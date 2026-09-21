"""Extract the recorded source token samples; fixed manifest is required."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import features as S

def indices(rows):
    legacy=sorted([r for r in rows if r['legacy_sampling']],key=lambda r:r['image'])
    rng=np.random.RandomState(42);lookup={r['image']:rng.choice(6400,2000,replace=False).astype(np.uint16) for r in legacy}
    for r in rows:
        if not r['legacy_sampling']:
            seed=int.from_bytes(hashlib.sha256(('42:'+r['image']).encode()).digest()[:4],'little')
            lookup[r['image']]=np.random.RandomState(seed).choice(6400,2000,replace=False).astype(np.uint16)
    return np.stack([lookup[r['image']] for r in rows])

def main():
    import torch
    import torch.nn.functional as F
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=str(Path(__file__).parent/'data/source_views14000.json'))
    ap.add_argument('--source-root',required=True);ap.add_argument('--dinov2-repo',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty directory; incomplete outputs are retained for diagnosis')
    rows=json.loads(Path(a.manifest).read_text());idx=indices(rows)
    b=json.loads((Path(__file__).parent/'models/model.json').read_text())
    if S.sha(a.checkpoint)!=b['encoder_sha256']:raise ValueError('Encoder hash mismatch')
    out.mkdir(parents=True,exist_ok=True);np.save(out/'indices.npy',idx)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dev=torch.device(a.device);enc=S.load_encoder(a.dinov2_repo,a.checkpoint,dev)
    cache=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16,shape=(len(rows),2000,384));done=np.zeros(len(rows),np.uint8)
    with torch.inference_mode():
        for i,r in enumerate(rows):
            p=Path(a.source_root)/r['image']
            if S.sha(p)!=r['sha256']:raise ValueError('Image bytes differ: '+r['image'])
            with Image.open(p) as im:boxed,_=S.letterbox(im)
            tokens=F.normalize(enc.forward_features(S.tensor(boxed,dev))['x_norm_patchtokens'][0].float(),dim=-1)
            cache[i]=tokens[torch.as_tensor(idx[i].astype(np.int64),device=dev)].half().cpu().numpy();done[i]=1
            if i%25==0 or i==len(rows)-1:cache.flush();np.save(out/'completed.npy',done);print(i+1,'/',len(rows),flush=True)
    (out/'identity.json').write_text(json.dumps(dict(manifest_sha256=S.sha(a.manifest),checkpoint_sha256=S.sha(a.checkpoint),feature_sha256=S.sha(out/'features.npy'),indices_sha256=S.sha(out/'indices.npy'),images=len(rows),target_GT_accessed=False),indent=2))

if __name__=='__main__':main()
