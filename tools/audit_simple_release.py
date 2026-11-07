"""Read-only dataset audit; writes summaries, hashes and record IDs, never raw data."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/simple-release-audit'

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def finite_pair(x):
    return isinstance(x, list) and len(x) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in x)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files, issues, records = [], [], {}
    field_presence = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    paths = collections.defaultdict(list)
    inputs = collections.defaultdict(list)
    raw_by_name = {}
    for f in sorted((ROOT / 'data/collision_dataset').glob('*.json')):
        d = json.loads(f.read_text())
        if 'sequences' in d:
            raw_by_name[f.name] = d
    diffs = collections.Counter()
    exclusions = []
    sensitive = collections.Counter()
    def scan(value):
        if isinstance(value, dict):
            for child in value.values(): scan(child)
        elif isinstance(value, list):
            for child in value: scan(child)
        elif isinstance(value, str):
            for name, pattern in [('google_key', r'AIza[\w-]{30,}'), ('local_path', r'/(?:Users|home|mnt|tmp)/'), ('email', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'), ('url', r'https?://'), ('private_key', r'BEGIN .*PRIVATE KEY')]:
                if re.search(pattern, value): sensitive[name] += 1
    for f in sorted((ROOT / 'LLM-RobotPath/collision_dataset').rglob('*.json')):
        d = json.loads(f.read_text())
        if not isinstance(d, dict) or 'sequences' not in d:
            continue
        m, seqs = d['metadata'], d['sequences']
        group = str(f.parent.relative_to(ROOT / 'LLM-RobotPath/collision_dataset'))
        mode = m.get('collection_config', {}).get('motion', {}).get('control_mode')
        info = dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size, group=group, records=len(seqs), collisions=sum(s.get('collision_detected') is True for s in seqs), mode=mode)
        files.append(info)
        groups[group].update(files=1, records=len(seqs), collisions=info['collisions'])
        if group not in ('.', 'Unseen'):
            continue
        scan(d)
        obs = m.get('warehouse_layout', {}).get('warehouse_geometry_model', {}).get('obstacles', [])
        layout = [{k:o.get(k) for k in ('element_id','center_point_m','dimensions_m','orientation_rad')} for o in obs]
        info['layout_hash'] = digest(layout)
        info['robot'] = m.get('robot_info', {}).get('id')
        info['metadata_count_matches'] = m.get('total_sequences') == len(seqs)
        info['asset_geometry'] = dict(robot_hull=bool(m.get('robot_info',{}).get('shape_info',{}).get('hull_3d')), collider_obstacles=sum(bool(o.get('collision_geometry')) for o in obs))
        original = raw_by_name.get(f.name)
        if original:
            diffs['matched_files'] += 1
            if original['metadata'] != m: diffs['metadata_changed_files'] += 1
            origseqs = {s['test_id']:s for s in original['sequences']}
            for s in seqs:
                old = origseqs.get(s['test_id'], {})
                for k in old.keys() | s.keys():
                    if old.get(k) != s.get(k): diffs['record_field:' + k] += 1
        seen = set()
        for s in seqs:
            uid = f.name + ':' + str(s.get('test_id'))
            flags = []
            if uid in seen: flags.append('duplicate_id')
            seen.add(uid)
            wp, hit, status = s.get('waypoint_sequence'), s.get('collision_detected'), s.get('navigation_status')
            eligible = s.get('training_eligible', True) is not False and (hit is True or status == 'success')
            if not eligible:
                exclusions.append(dict(id=uid, reason=s.get('collection_outcome'), navigation_status=status, collision=hit))
            groups[group]['eligible'] += eligible
            groups[group]['eligible_collisions'] += eligible and hit is True
            groups[group]['status:' + str(status)] += 1
            groups[group]['training_eligible:' + str(s.get('training_eligible','missing'))] += 1
            field_presence.update(s.keys())
            if not isinstance(wp,list) or len(wp)<2 or not all(finite_pair(p) for p in wp): flags.append('invalid_waypoints')
            if s.get('sequence_length') != len(wp or []): flags.append('sequence_length_mismatch')
            if type(hit) is not bool: flags.append('invalid_collision_label')
            if not hit and status != 'success': flags.append('incomplete_noncollision')
            if hit:
                if status != 'collision': flags.append('collision_status_mismatch')
                if not finite_pair(s.get('collision_position')): flags.append('invalid_collision_position')
                idx = s.get('collision_waypoint_index')
                if type(idx) is not int or not 1 <= idx < len(wp): flags.append('invalid_collision_index')
                if not s.get('collision_object_id'): flags.append('missing_collision_object_id')
                if isinstance(idx, int) and s.get('collision_segment') != [idx-1,idx]: flags.append('segment_convention_mismatch')
            tr = s.get('executed_trajectory', [])
            if not tr: flags.append('missing_trajectory')
            if any(abs(t.get('commanded_linear_speed',0))>1e-6 and abs(t.get('commanded_angular_speed',0))>1e-6 for t in tr): flags.append('simultaneous_commands')
            if any(b.get('simulation_time',0)<a.get('simulation_time',0) for a,b in zip(tr,tr[1:])): flags.append('nonmonotonic_trajectory_time')
            if any(abs(v)>6 for p in (wp or []) for v in p): flags.append('waypoint_outside_12m_bounds')
            if flags: issues.append(dict(id=uid, group=group, eligible=eligible, flags=flags))
            records[uid] = dict(group=group, eligible=eligible, collision=hit, layout=info['layout_hash'], robot=info['robot'])
            paths[digest(wp)].append(uid)
            inputs[digest([m.get('robot_info'), layout, wp])].append(uid)
    manifest_path = ROOT / 'LLM-RobotPath/outputs/audit_20260919/split_manifest.json'
    split = json.loads(manifest_path.read_text())
    sets = {k:{f'{name}:{i}' for name,i in split[k]} for k in ('train','validation','test')}
    union = set.union(*sets.values())
    eligible_main = {k for k,v in records.items() if v['group']=='.' and v['eligible']}
    split_audit = dict(counts={k:len(v) for k,v in sets.items()}, missing_ids=sorted(union-records.keys()), ineligible_ids=sorted(union-eligible_main), eligible_main_not_in_split=sorted(eligible_main-union), overlaps={f'{a}/{b}':len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]})
    split_audit['layout_overlap_train_unseen'] = len({records[k]['layout'] for k in sets['train'] if k in records} & {v['layout'] for v in records.values() if v['group']=='Unseen'})
    split_audit['exact_path_overlap_train_unseen'] = sum(any(k in sets['train'] for k in v) and any(records[k]['group']=='Unseen' for k in v) for v in paths.values())
    summary = dict(sensitive_pattern_hits=dict(sensitive), excluded_reasons=dict(collections.Counter(x['reason'] for x in exclusions)), groups=dict(groups), raw_vs_enriched=dict(diffs), simple_fields_present=dict(field_presence), flags=dict(collections.Counter(flag for issue in issues for flag in issue['flags'])), eligible_flag_counts=dict(collections.Counter(flag for issue in issues if issue['eligible'] for flag in issue['flags'])), duplicate_path_groups=sum(len(v)>1 for v in paths.values()), duplicate_input_groups=sum(len(v)>1 for v in inputs.values()), layouts={g:len({v['layout'] for v in records.values() if v['group']==g}) for g in ('.','Unseen')}, robots=sorted({str(v['robot']) for v in records.values()}), split_audit=split_audit)
    for name,obj in [('summary',summary),('source_manifest',files),('record_issues',issues),('excluded_records',exclusions)]:
        (OUT / (name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
