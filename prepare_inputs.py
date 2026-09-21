"""Create a GT-free, deterministic single-image/folder inference manifest."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.input);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    files=[src] if src.is_file() else sorted(x for x in src.rglob('*') if x.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff'] and x.is_file())
    if not files:raise ValueError('No images')
    rows=[dict(image_id=x.stem,image=str(x.resolve())) for x in files]
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate filename stems; supply a manifest with unique IDs')
    out.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
