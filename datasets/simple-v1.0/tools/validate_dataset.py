"""Verify checksums, schema and partition integrity; no simulator/model needed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from load_dataset import DEFAULT_ROOT, read_records

EXPECTED = {'train':4903,'validation':610,'test':617,'layout_transfer':300,'robot_transfer':300,'joint_transfer':300}

def require(condition, message):
    if not condition: raise ValueError(message)

def pair(v):
    return isinstance(v,list) and len(v)==2 and all(type(n) in (int,float) and math.isfinite(n) for n in v)

def polygon(p):
    return isinstance(p,list) and len(p)>=3 and all(pair(v) for v in p)

def validate(root):
    root = Path(root).resolve()
    checks = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ',1)
        path = (root/name).resolve()
        require(path.is_relative_to(root), 'Unsafe checksum path')
        require(name not in checks,'Duplicate checksum entry')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,f'Checksum mismatch: {name}')
        checks[name] = digest
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
    require(set(checks)==actual,'Checksum inventory differs from package files')
    inputs = read_records(root/'inputs/records.jsonl')
    labels = read_records(root/'labels/records.jsonl')
    scenes = json.loads((root/'scenes.json').read_text())
    splits = json.loads((root/'splits.json').read_text())
    require(set(inputs)==set(labels) and len(inputs)==7030,'Input/label IDs or count mismatch')
    require(set(splits)==set(EXPECTED),'Unexpected split names')
    ids = [sid for group in splits.values() for sid in group]
    require(len(ids)==len(set(ids)) and set(ids)==set(inputs),'Splits overlap or omit records')
    for name,n in EXPECTED.items(): require(len(splits[name])==n,f'Incorrect split size: {name}')
    for scene in scenes.values():
        for o in scene['obstacles']:
            require(pair(o['center_point_m']),'Invalid obstacle center')
            if 'obstacle_polygon_m' in o: require(polygon(o['obstacle_polygon_m']),'Invalid obstacle polygon')
    for sid,x in inputs.items():
        y = labels[sid]
        require(set(x)=={'sample_id','scene_id','motion_mode','robot','waypoints_m'},'Unexpected input field')
        require(x['scene_id'] in scenes and x['motion_mode']=='simple','Invalid scene or mode')
        require(len(x['waypoints_m'])>=2 and all(pair(p) for p in x['waypoints_m']),'Invalid path')
        shape=x['robot']['footprint_local_m']
        components=shape.get('polygon_components') or [shape.get('polygon')]
        require(all(polygon(p) for p in components),'Invalid footprint')
        require(type(y['collision']) is bool,'Invalid collision flag')
        require(y['collision_fields_valid'] is y['collision'],'Invalid field mask')
        if y['collision']:
            i=y['collision_waypoint_index']
            require(type(i) is int and 1<=i<len(x['waypoints_m']),'Invalid index')
            require(y['collision_segment']==[i-1,i],'Invalid segment')
            require(pair(y['collision_position_m']),'Invalid contact position')
            require(isinstance(y['collision_object_id'],str) and bool(y['collision_object_id']),'Missing object')
            require(pair(y['robot_center_at_collision_m']),'Invalid robot center')
        else:
            require(all(y[k] is None for k in ('collision_waypoint_index','collision_segment','collision_position_m','collision_object_id','robot_center_at_collision_m')),'Non-null negative target')
    require(sum(y['collision'] for y in labels.values())==4745,'Incorrect positive count')
    excluded=json.loads((root/'exclusions.json').read_text())
    require(len(excluded)==170 and len({x['sample_id'] for x in excluded})==170,'Invalid exclusions')
    require(not set(inputs)&{x['sample_id'] for x in excluded},'Excluded sample leaked')
    for row in json.loads((root/'examples.json').read_text()):
        sid=row['input']['sample_id']
        require(sid in splits[row['split']] and row['input']==inputs[sid] and row['label']==labels[sid],'Example mismatch')
    prov=json.loads((root/'provenance.json').read_text())
    require(prov['records']==len(inputs),'Provenance count mismatch')
    for name,group in splits.items():
        require(prov['splits'][name]=={'records':len(group),'collision':sum(labels[s]['collision'] for s in group)},'Provenance split mismatch')
    return {'valid':True,'records':len(inputs),'collisions':4745,'noncollisions':2285,'splits':EXPECTED}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    try: print(json.dumps(validate(args.data_dir),indent=2))
    except (ValueError,KeyError,TypeError,OSError) as exc: parser.exit(1,f'Validation failed: {exc}\n')
