"""Compare a short 30-fps frame-timecode trim with its decimal equivalent.

Supply original NRD, TSD and FM ZIPs yourself. No recordings are distributed.
The retained range is only a verification example, not the user's final cut.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from oculostudio.storage import ProjectStore
from oculostudio.services import add_asset
from oculostudio.services.native_reference_trim import preview_native_reference_trim, export_native_reference_trim


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',required=True);ap.add_argument('--tsd',required=True)
    ap.add_argument('--facemapper',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args()
    sources={k:Path(v).resolve() for k,v in [('native',a.native),('tsd',a.tsd),('facemapper',a.facemapper)]}
    hashes={k:sha(v) for k,v in sources.items()}
    with tempfile.TemporaryDirectory(prefix='oculo_timecode_real_') as tmp:
        store=ProjectStore.create(Path(tmp)/'project','Timecode equivalence verification')
        sid=store.project.selected_session_id
        ids={k:add_asset(store,session_id=sid,path=v,checksum=True).id for k,v in sources.items()}
        req=dict(session_id=sid,native_asset_id=ids['native'],timeseries_asset_id=ids['tsd'],
                 facemapper_asset_id=ids['facemapper'],label='Verification-only short interval')
        pa=preview_native_reference_trim(store,**req,start_timecode='00:00:17:03',duration_timecode='00:00:07:15',timecode_fps=30)
        pb=preview_native_reference_trim(store,**req,start_timecode='17.1',duration_timecode='7.5')
        assert pa['utc_start_ns']==pb['utc_start_ns'] and pa['utc_end_ns']==pb['utc_end_ns']
        assert pa['report']==pb['report']
        ra=Path(export_native_reference_trim(store,session_id=sid,preview_id=pa['preview_id'])['output_dir'])
        rb=Path(export_native_reference_trim(store,session_id=sid,preview_id=pb['preview_id'])['output_dir'])
        results=[]
        for csv in sorted(ra.rglob('*.csv')):
            rel=csv.relative_to(ra)
            assert csv.read_bytes()==(rb/rel).read_bytes(),rel
            results.append({'file':rel.as_posix(),'identical_bytes':True})
        unchanged=all(sha(v)==hashes[k] for k,v in sources.items())
        assert unchanged
        result={'release':'0.5.4','status':'passed','scope':'One short I05 real-data equivalence test; not the user\'s I01 final cut.',
                'inputs':[{'role':k,'file':v.name,'sha256':hashes[k]} for k,v in sources.items()],
                'frame_input':{'start':'00:00:17:03','duration':'00:00:07:15','timecode_fps':30},
                'equivalent_seconds':{'start':'17.1','duration':'7.5'},
                'resolved_start':pa['native_start'],'resolved_end':pa['native_end'],
                'native_decoded_frames':pa['native_video_frame_count'],
                'same_utc_window':True,'same_preview_counts':True,
                'gaze_rows':pa['report']['gaze_rows'],'fm_exact_gaze_matches':pa['report']['fm_exact_gaze_matches'],
                'csv_files':results,'csv_count':len(results),'original_archives_unchanged':unchanged,
                'videos_reencoded':False,'recording_data_bundled':False}
    Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
