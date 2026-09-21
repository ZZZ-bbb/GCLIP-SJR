"""Source-only semantic assignment; never accepts target masks or target scores."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import features as S

def choose(margins):
    values=np.asarray(margins,dtype=np.float64)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Every calibration image and component must have a finite score')
    means=values.mean(0);order=np.argsort(-means,kind='stable')
    return dict(selected_component=int(order[0]),mean_semantic_margin=means.tolist(),
                top_two_margin=float(means[order[0]]-means[order[1]]))

def embedding(x):
    import torch
    return x if isinstance(x,torch.Tensor) else x.pooler_output

def main():
    import torch
    import torch.nn.functional as F
    import controlled_source_clustering as C
    ap=argparse.ArgumentParser();base=Path(__file__).parent
    ap.add_argument('--source-root',required=True);ap.add_argument('--selection',default=str(base/'data/calibration20.json'))
    ap.add_argument('--model',default=str(base/'models/model.json'));ap.add_argument('--prompts',default=str(base/'configs/clip_prompts.json'))
    ap.add_argument('--dinov2-repo',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--clip-directory',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    b=read(a.model);selection=read(a.selection);cfg=read(a.prompts);mp=Path(a.model).parent/b['model_file']
    if S.sha(mp)!=b['model_sha256'] or S.sha(a.checkpoint)!=b['encoder_sha256']:raise ValueError('Model hash mismatch')
    if selection['checkpoint_sha256']!=b['encoder_sha256']:raise ValueError('Calibration encoder mismatch')
    model=C.FrozenClusterModel.load(mp);device=torch.device(a.device)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    encoder=S.load_encoder(a.dinov2_repo,a.checkpoint,device)
    from transformers import CLIPModel,CLIPConfig,CLIPProcessor
    md=Path(a.clip_directory);clip=CLIPModel(CLIPConfig.from_pretrained(md,local_files_only=True))
    state=torch.load(md/'pytorch_model.bin',map_location='cpu',weights_only=True)
    extra=set(state)-set(clip.state_dict())
    if not all(k.endswith('.position_ids') for k in extra):raise ValueError('Unexpected checkpoint keys')
    for k in extra:del state[k]
    clip.load_state_dict(state,strict=True);clip.eval().requires_grad_(False).to(device)
    processor=CLIPProcessor.from_pretrained(md,local_files_only=True,use_fast=False)
    recorded=read(base/'data/clip_assignment_scores.json')['clip_files']['files']
    for name,record in recorded.items():
        if S.sha(md/name)!=record['sha256']:raise ValueError('CLIP asset mismatch: '+name)
    out.mkdir(parents=True,exist_ok=True);records=[]
    with torch.inference_mode():
        prompts=[s for name in ['panicle','leaf','background'] for s in cfg['text_classes'][name]]
        text=embedding(clip.get_text_features(**processor(text=prompts,padding=True,return_tensors='pt').to(device)))
        prototypes=F.normalize(F.normalize(text,dim=-1).reshape(3,3,-1).mean(1),dim=-1)
        for item in selection['images']:
            p=Path(a.source_root)/item['path']
            if S.sha(p)!=item['sha256']:raise ValueError('Source image hash mismatch: '+str(p))
            with Image.open(p) as im:boxed,pad=S.letterbox(im)
            if pad[:4]!=(0,0,1120,1120):raise ValueError('Recorded calibration protocol requires square source images')
            labels=S.classify(S.dense_features(encoder,boxed,device),{'model':model},device)['model']
            views=[];counts=[]
            for k in range(model.components):
                mask=labels==k;counts.append(int(mask.sum()))
                if not counts[-1]:raise ValueError('Empty component; abort without imputation or skipping')
                view=np.full_like(boxed,cfg['neutral_rgb']);view[mask]=boxed[mask];views.append(Image.fromarray(view))
            image=embedding(clip.get_image_features(**processor(images=views,return_tensors='pt').to(device)))
            cos=(F.normalize(image,dim=-1)@prototypes.T).cpu().numpy();margins=cos[:,0]-cos[:,1:].max(1)
            records.append(dict(path=item['path'],sha256=item['sha256'],component_pixels=counts,class_cosines=cos.tolist(),semantic_margin=margins.tolist()))
            print(len(records),'/',len(selection['images']),flush=True)
    assignment=choose([r['semantic_margin'] for r in records]);assignment.update(records=records,target_data_accessed=False,model_sha256=S.sha(mp),selection_sha256=S.sha(a.selection),prompts_sha256=S.sha(a.prompts))
    dest=out/'assignment.json';dest.write_text(json.dumps(assignment,indent=2))
    binding=dict(b,foreground_components=[assignment['selected_component']],assignment_record_sha256=S.sha(dest),assignment_file='assignment.json')
    # Keep the relative model reference portable within the output package.
    import shutil
    shutil.copy2(mp,out/mp.name);binding['model_file']=mp.name
    (out/'model.json').write_text(json.dumps(binding,indent=2))

if __name__=='__main__':main()
