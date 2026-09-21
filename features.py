"""Frozen DINOv2 feature and spatial protocol; no CLIP or target masks."""
from pathlib import Path
import hashlib
import json
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def letterbox(im,size=1120):
    import cv2
    from PIL import ImageOps
    rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    h,w=rgb.shape[:2];scale=min(size/h,size/w)
    nh,nw=max(1,int(round(h*scale))),max(1,int(round(w*scale)))
    resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top,left=(size-nh)//2,(size-nw)//2
    return cv2.copyMakeBorder(resized,top,size-nh-top,left,size-nw-left,cv2.BORDER_CONSTANT,value=(0,0,0)),(top,left,nh,nw,h,w)

def tensor(boxed,device):
    import torch
    x=torch.from_numpy(np.ascontiguousarray(boxed)).permute(2,0,1).float().div_(255.)
    return ((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])[None].to(device)

def load_encoder(repo,checkpoint,device):
    import torch
    verify_encoder_source(repo)
    model=torch.hub.load(str(Path(repo).resolve()),'dinov2_vits14',source='local',pretrained=False,verbose=False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device)

def verify_encoder_source(repo):
    cfg=json.loads((Path(__file__).parent/'configs/dinov2_source.json').read_text())
    for name,expected in cfg['files'].items():
        if sha(Path(repo)/name)!=expected:
            raise ValueError('DINOv2 source identity mismatch: '+name)

def dense_features(encoder,boxed,device):
    import torch.nn.functional as F
    tokens=F.normalize(encoder.forward_features(tensor(boxed,device))['x_norm_patchtokens'][0].float(),dim=-1)
    # Preserve the original float16 cache round trip, then classify dense float32 features.
    tokens=tokens.half().cpu().numpy()
    import torch
    grid=torch.as_tensor(tokens.copy(),device=device).float().T.reshape(1,384,80,80)
    return F.normalize(F.interpolate(grid,size=(1120,1120),mode='bilinear',align_corners=False),dim=1)

def classify(dense,models,device,probability=False):
    import torch
    import controlled_source_clustering as C
    params={n:C.parameters(m,device) for n,m in models.items()}
    out={n:np.empty((1120,1120,m.components),np.float32) if probability else np.empty((1120,1120),np.uint8) for n,m in models.items()}
    for start in range(0,1120,64):
        end=min(start+64,1120);x=dense[0,:,start:end].permute(1,2,0).reshape(-1,384)
        for n,p in params.items():
            scores=C.scores(x,p)
            out[n][start:end]=(scores.softmax(1).reshape(end-start,1120,-1).cpu().numpy() if probability else scores.argmax(1).reshape(end-start,1120).cpu().numpy().astype(np.uint8))
    return out
