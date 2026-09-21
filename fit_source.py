"""Fit a controlled source model from a (views, 2000, 384) float16 NPY cache.

The cache view order must match the supplied source manifest. This script does
not build semantic identity or access target labels. Use the released fixed
models for exact replay of the reported experiments.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import controlled_source_clustering as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(a.features,mmap_mode='r',allow_pickle=False)
    if Path(a.output).exists() or Path(a.output+'.json').exists():raise FileExistsError('Never overwrite an existing fit')
    minimum=500
    if data.ndim!=3 or data.shape[1:]!=(2000,384) or data.shape[0]<minimum:raise ValueError('Expected at least 500 views × 2000 tokens × 384 features')
    rng=np.random.default_rng(3407);ids=rng.choice(len(data),min(len(data),500),replace=False)
    sample=np.concatenate([np.asarray(data[i,rng.choice(2000,400,replace=False)],np.float32) for i in ids])
    dev=torch.device(a.device);kind='gmm';cov='diag'
    model,init=C.initialize(sample,kind,cov,3,a.seed,dev,65536,1e-6)
    model,report=C.fit(data.reshape(-1,384),model,dev,65536,100,.001,1e-6)
    model.save(a.output)
    Path(a.output+'.json').write_text(json.dumps({'initialization':init,'fit':report},indent=2),encoding='utf-8')

if __name__=='__main__':main()
