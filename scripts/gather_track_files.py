import glob
import json

all_files = [file for file in glob.glob("dino_tracker_vis/run_*_traj_mixp_object_new_v4.json") if 'all' not in file]
all_files.sort()

all_trjs = []
for fid, file in enumerate(all_files):
    for trj in json.load(open(file)):
        trj['total_frame_idx'] = len(all_trjs)
        trj['parquet_idx'] = fid
        all_trjs.append(trj)

json.dump(all_trjs, open("dino_tracker_vis/run_all_traj_mixp_object_new_v4.json" ,"w"))
        