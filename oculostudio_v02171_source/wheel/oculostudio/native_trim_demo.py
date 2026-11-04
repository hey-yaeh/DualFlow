"""Small, fully synthetic Native/TSD/multi-recording FM example. No mocap needed."""
from __future__ import annotations
import csv
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
from oculostudio.storage import ProjectStore
from oculostudio.services import add_asset
from oculostudio.services.native_reference_trim import preview_native_reference_trim, export_native_reference_trim
from oculostudio.services.pretrim_overlay import render_pretrim_overlay_preview


def create_native_trim_demo(root: str | Path) -> dict:
    root=Path(root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Demo folder is not empty; choose a new directory')
    store=ProjectStore.create(root,'Native-reference trim · Synthetic demo')
    data=root/'synthetic_inputs'; data.mkdir(parents=True,exist_ok=True)
    video=data/'synthetic_scene.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(160,120))
    if not writer.isOpened():raise RuntimeError('Synthetic video encoder is unavailable')
    for i in range(120):
        image=np.full((120,160,3),[48,65,72],dtype=np.uint8)
        cv2.circle(image,(80,60),25,(160,195,180),-1)
        cv2.putText(image,str(i),(8,108),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
        writer.write(image)
    writer.release()
    base=1_800_000_000_000_000_097
    rid='synthetic-native-reference';other='synthetic-other-recording'
    times=np.array([base+round(i/30*1e9) for i in range(120)],dtype='<i8')
    def table(cols,rows):
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(cols);w.writerows(rows);return s.getvalue()
    nrd=data/'Synthetic_NRD.zip';tsd=data/'Synthetic_TSD.zip';fm=data/'Synthetic_FaceMapper.zip'
    info=json.dumps({'recording_id':rid,'start_time':base-2_500_000_000,'duration':6_500_000_000})
    with zipfile.ZipFile(nrd,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('recording/info.json',info)
        z.writestr('recording/Neon Scene Camera v1 ps1.mp4',video.read_bytes())
        z.writestr('recording/Neon Scene Camera v1 ps1.time',times.tobytes())
    video.unlink()
    gaze_times=[base+i*5_000_000 for i in range(800)]
    section_cols=['section id','recording id','section start time [ns]','section end time [ns]']
    gaze_cols=['section id','recording id','timestamp [ns]','gaze x [px]','gaze y [px]','worn']
    with zipfile.ZipFile(tsd,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('recording/info.json',info)
        z.writestr('recording/gaze.csv',table(gaze_cols,[['TSD-section',rid,t,'80.01234','55.45678','1'] for t in gaze_times]))
        z.writestr('recording/3d_eye_states.csv',table(['recording id','timestamp [ns]','pupil diameter left [mm]'],[[rid,t,'3.75'] for t in gaze_times]))
        z.writestr('recording/world_timestamps.csv',table(['recording id','timestamp [ns]'],[[rid,int(t)] for t in times]))
        z.writestr('recording/fixations.csv',table(['recording id','fixation id','start timestamp [ns]','end timestamp [ns]','duration [ms]','fixation x [px]'],[[rid,1,base+100_000_000,base+1_000_000_000,900,'80.01'],[rid,2,base+2_000_000_000,base+3_000_000_000,1000,'79.9']]))
        z.writestr('sections.csv',table(section_cols,[['TSD-section',rid,base,base+4_000_000_000]]))
    with zipfile.ZipFile(fm,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('sections.csv',table(section_cols,[['other-section',other,base,base+4_000_000_000],['FM-section',rid,base,base+4_000_000_000]]))
        z.writestr('gaze_on_face.csv',table(['section id','recording id','timestamp [ns]','gaze on face'],[['other-section',other,t,'False'] for t in gaze_times]+[['FM-section',rid,t,'True'] for t in gaze_times]))
        z.writestr('face_detections.csv',table(['recording id','timestamp [ns]','p1 x [px]','p1 y [px]','p2 x [px]','p2 y [px]','eye left x [px]','eye left y [px]','eye right x [px]','eye right y [px]','nose x [px]','nose y [px]','mouth left x [px]','mouth left y [px]','mouth right x [px]','mouth right y [px]'],[[rid,int(t)-100,55,35,105,90,70,50,90,50,80,60,72,75,88,75] for t in times]))
    sid=store.project.selected_session_id
    ids={role:add_asset(store,session_id=sid,path=path,label=label,checksum=True).id for role,path,label in [('native',nrd,'Native scene recording'),('tsd',tsd,'Eye-tracking timeseries'),('face_mapper',fm,'Face Mapper · two recordings')]}
    preview=preview_native_reference_trim(store,session_id=sid,native_asset_id=ids['native'],timeseries_asset_id=ids['tsd'],facemapper_asset_id=ids['face_mapper'],start_timecode='00:00:00.500',duration_timecode='00:00:02.000',label='Synthetic 0.5s + 2s')
    overlay=render_pretrim_overlay_preview(store,session_id=sid,preview_id=preview['preview_id'],scope='representative',segment_seconds=1.0,output_fps=15)
    result=export_native_reference_trim(store,session_id=sid,preview_id=preview['preview_id'])
    return {'project':str(store.manifest_path),'assets':ids,'session_id':sid,'overlay_preview':overlay,**result}
