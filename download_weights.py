"""Explicit opt-in downloads from official model endpoints, with SHA256 checks."""
import argparse,io,json,urllib.request,zipfile
from pathlib import Path
from features import sha

def get_source(destination):
    """Download a pinned official archive; validate every archived research Python file."""
    cfg=json.loads((Path(__file__).parent/'configs/dinov2_source.json').read_text())
    root=Path(destination)/('dinov2-'+cfg['commit'])
    if not root.exists():
        url=f"https://codeload.github.com/facebookresearch/dinov2/zip/{cfg['commit']}"
        with urllib.request.urlopen(url,timeout=60) as response:blob=response.read(10*1024*1024+1)
        if len(blob)>10*1024*1024:raise ValueError('Unexpected source archive size')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for entry in archive.infolist():
                target=(Path(destination)/entry.filename).resolve()
                if not target.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
                if ((entry.external_attr>>16)&0o170000)==0o120000:raise ValueError('Archive symlink rejected')
            archive.extractall(destination)
    for name,expected in cfg['files'].items():
        if sha(root/name)!=expected:raise ValueError('Source file hash mismatch: '+name)
    print('Verified official source: '+str(root))
    return root

def get(url,path,expected):
    if path.exists():
        if sha(path)!=expected:raise ValueError('Existing file hash mismatch; not overwritten: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
    if part.exists():raise FileExistsError('Existing partial download: '+str(part))
    with urllib.request.urlopen(url,timeout=60) as inp,part.open('wb') as out:
        for block in iter(lambda:inp.read(1024*1024),b''):out.write(block)
    if sha(part)!=expected:raise ValueError('Download checksum mismatch; partial retained')
    part.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--clip',action='store_true');ap.add_argument('--source-only',action='store_true',help='Obtain/verify only the 3 MB pinned official DINOv2 source');a=ap.parse_args()
    base=Path(__file__).parent;out=Path(a.output);binding=json.loads((base/'models/model.json').read_text())
    get_source(out)
    if not a.source_only:get('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',out/'dinov2_vits14_pretrain.pth',binding['encoder_sha256'])
    if a.clip:
        cfg=json.loads((base/'configs/clip_prompts.json').read_text());records=json.loads((base/'data/clip_assignment_scores.json').read_text())['clip_files']['files']
        for name,r in records.items():get(f"https://huggingface.co/{cfg['clip_model']}/resolve/{cfg['clip_revision']}/{name}",out/'clip'/name,r['sha256'])
    print('Requested source/model assets verified.')

if __name__=='__main__':main()
