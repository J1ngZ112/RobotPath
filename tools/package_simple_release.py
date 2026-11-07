"""Create checksums and a reproducible standalone archive from the release tree."""
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'datasets/simple-v1.0'

def main():
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    checks = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(PACKAGE)}\n' for p in files)
    (PACKAGE/'SHA256SUMS').write_text(checks)
    files.append(PACKAGE/'SHA256SUMS')
    out = ROOT/'downloads/robotpath-simple-v1.0.tar.gz'
    out.parent.mkdir(exist_ok=True)
    with out.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for p in sorted(files):
                    raw = p.read_bytes()
                    item = tarfile.TarInfo('robotpath-simple-v1.0/'+str(p.relative_to(PACKAGE)))
                    item.size=len(raw); item.mode=0o644; item.mtime=0
                    tar.addfile(item,io.BytesIO(raw))
    out.with_suffix(out.suffix+'.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
    print(f'{out.relative_to(ROOT)}: {out.stat().st_size:,} bytes')

if __name__=='__main__':main()
