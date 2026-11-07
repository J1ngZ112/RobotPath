"""Standard-library loader. Import iter_samples or run for a small preview."""
import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]

def read_records(path):
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            sid = row['sample_id']
            if sid in records:
                raise ValueError(f'Duplicate sample ID: {sid}')
            records[sid] = row
    return records

def iter_samples(root=DEFAULT_ROOT, split='test'):
    root = Path(root)
    splits = json.loads((root/'splits.json').read_text())
    if split not in splits:
        raise ValueError(f'Unknown split {split!r}; choose from {list(splits)}')
    inputs = read_records(root/'inputs/records.jsonl')
    labels = read_records(root/'labels/records.jsonl')
    scenes = json.loads((root/'scenes.json').read_text())
    for sid in splits[split]:
        yield {'input':inputs[sid], 'scene':scenes[inputs[sid]['scene_id']], 'label':labels[sid]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0: parser.error('--limit must be nonnegative')
    for i,row in enumerate(iter_samples(args.data_dir,args.split)):
        if i >= args.limit: break
        print(json.dumps({'sample_id':row['input']['sample_id'],'robot':row['input']['robot']['id'],'waypoints':len(row['input']['waypoints_m']),'collision':row['label']['collision']},ensure_ascii=False))

if __name__=='__main__':main()
