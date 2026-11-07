"""Export the approved Simple snapshot without model code or raw asset files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/simple-v1.0'
AUDIT = ROOT / 'docs/simple-release-audit'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def ident(value):
    return sha(json.dumps(value, sort_keys=True).encode())[:20]

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    plan = json.loads((AUDIT/'candidate_splits.json').read_text())
    source_splits = {**plan['splits'], **plan['transfer']}
    manifest = json.loads((AUDIT/'source_manifest.json').read_text())
    inputs, labels, scenes, sources, excluded = {}, {}, {}, [], []
    uid_map = {}
    private_map = {}
    counts = {}
    for spec in manifest:
        if spec['group'] not in ('.','Unseen'):
            continue
        f = ROOT/spec['path']
        raw = f.read_bytes()
        if sha(raw) != spec['sha256']:
            raise ValueError(f'Snapshot changed: {f.name}; rerun audit before exporting')
        d = json.loads(raw)
        m = d['metadata']
        batch_id = 'batch_'+ident(f.name)
        sources.append({'batch_id':batch_id,'sha256':sha(raw),'raw_records':len(d['sequences'])})
        model = m['warehouse_layout']['warehouse_geometry_model']
        obstacles = [{k:o[k] for k in ('element_id','center_point_m','dimensions_m','orientation_rad','obstacle_polygon_m') if k in o} for o in model['obstacles']]
        scene = {'warehouse_size_m':model['global_summary']['warehouse_size_m'], 'obstacles':obstacles}
        scene_id = 'scene_'+ident(scene)
        scenes[scene_id] = scene
        robot = m['robot_info']
        shape = robot.get('shape_info', {})
        geometry = {k:shape[k] for k in ('polygon','polygon_components') if k in shape}
        if not geometry:
            raise ValueError('Missing original robot footprint')
        for s in d['sequences']:
            key = (f.name,s['test_id'])
            sid = 'simple_'+ident(key)
            if sid in private_map:
                raise ValueError('Duplicate sample id')
            private_map[sid] = {'source_file':str(f.relative_to(ROOT)), 'test_id':s['test_id']}
            uid_map[key] = sid
            if s.get('training_eligible',True) is False:
                excluded.append({'sample_id':sid,'reason':s.get('collection_outcome','training_ineligible')})
                continue
            if not s['collision_detected'] and s['navigation_status'] != 'success':
                raise ValueError('Unresolved incomplete negative')
            inputs[sid] = {'sample_id':sid, 'scene_id':scene_id, 'motion_mode':'simple', 'robot':{'id':robot['id'],'size_m':robot['size'],'footprint_local_m':geometry}, 'waypoints_m':s['waypoint_sequence']}
            hit = s['collision_detected']
            labels[sid] = {'sample_id':sid,'collision':hit,
                'collision_waypoint_index':s['collision_waypoint_index'] if hit else None,
                'collision_segment':s['collision_segment'] if hit else None,
                'collision_position_m':s['collision_position'] if hit else None,
                'collision_object_id':s['collision_object_id'] if hit else None,
                'robot_center_at_collision_m':s.get('robot_center_at_collision') if hit else None,
                'collision_fields_valid':hit}
    splits = {k:[uid_map[tuple(v)] for v in vals] for k,vals in source_splits.items()}
    flat = [sid for ids in splits.values() for sid in ids]
    if len(flat)!=len(set(flat)) or set(flat)!=set(inputs):
        raise ValueError('Splits do not partition exported data')
    for group, rows in [('inputs',inputs),('labels',labels)]:
        (OUT/group).mkdir(parents=True,exist_ok=True)
        with (OUT/group/'records.jsonl').open('w') as stream:
            for sid in sorted(rows): stream.write(json.dumps(rows[sid],ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    write(OUT/'scenes.json',scenes)
    write(OUT/'splits.json',splits)
    write(OUT/'exclusions.json',excluded)
    examples = []
    for hit in (True,False):
        sid = next(s for s in splits['train'] if labels[s]['collision']==hit)
        examples.append({'split':'train','input':inputs[sid],'label':labels[sid]})
    write(OUT/'examples.json',examples)
    for k,ids in splits.items(): counts[k]={'records':len(ids),'collision':sum(labels[i]['collision'] for i in ids)}
    write(OUT/'provenance.json', {'release':'simple-v1.0','schema_version':1,'records':len(inputs),'collisions':sum(v['collision'] for v in labels.values()),'excluded_records':len(excluded),'splits':counts,'source_batches':sources,'motion_control_mode':'rotate_then_translate_accumulated_pose_servo','annotation_validation':'Confirmed correct by the dataset author; export checks do not rerun simulation.','geometry_permission':'Author confirmed redistribution permission on 2026-11-07; supporting evidence to be supplied.','transformations':['Stable SHA-256-derived sample IDs; internal source mapping retained locally only.','Original numeric paths, footprints, obstacle polygons and positive labels preserved.','Negative collision-only fields normalized to null.','USD files, 3D hulls, per-part colliders, runtime diagnostics, generation hints and trajectories omitted.'],'split_policy':'Existing fixed main split preserved; three separate transfer tests; no new random split.'})
    write(ROOT/'outputs/simple-release-private/source_map.json',private_map)
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
