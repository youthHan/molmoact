import glob
import os

root_dir = "/mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/experiments/libero/rollouts/2025_09_13/libero_spatial/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/molmoact/allenai/MolmoAct-7B-D-LIBERO-Spatial-0812"
all_videos = glob.glob(root_dir+'/*mp4')

tasks_to_videos = {}
for vid in all_videos:
    task = vid.split('task=')[-1]
    status = vid.split('success=')[1].startswith('True')
    tasks_to_videos.setdefault(task, {}).setdefault(str(status), []).append(vid)

final_selection = []
for task, status_vids in tasks_to_videos.items():
    for status, vids in status_vids.items():
        final_selection.append(vids[0])

cp_cmd = [f"cp {path} /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/experiments/libero/rollouts/official_inference/spatial/" for path in final_selection]
[os.system(cmd) for cmd in cp_cmd]