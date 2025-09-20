sudo apt update
sudo apt install libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6 
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/

export PATH=/mnt/bn/kinetics-lp-maliva/envs/sum-cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/bn/kinetics-lp-maliva/envs/sum-cuda-12.4/lib64:$LD_LIBRARY_PATH


sudo chown -R tiger:tiger /opt/conda/
source /opt/conda/bin/activate 
cd ../LIBERO
/opt/conda/bin/pip install -e .
/opt/conda/bin/pip install easydict==1.9 einops==0.4.1 robosuite==1.4.0 bddl==1.0.1 future==0.18.2 matplotlib==3.5.3 cloudpickle==2.1.0 gym==0.25.2 tensorflow==2.15.0
/opt/conda/bin/pip install transformers==4.52.1 vllm==0.8.5

export VLLM_WORKER_MULTIPROC_METHOD=spawn

rm -rf ~/.local/lib/python3.11/site-packages/google/protobuf ~/.local/lib/python3.11/site-packages/protobuf-*.dist-info || true


/mnt/bn/kinetics-lp-maliva/envs/conda/envs/bagel/bin/torchrun --nproc_per_node=8 --master_port=19327 /mnt/bn/kinetics-lp-maliva-v6/tools/occu_full.py 


/mnt/bn/kinetics-lp-maliva/envs/conda/envs/bagel/bin/torchrun --nproc_per_node=8 --master_port=19367 /mnt/bn/kinetics-lp-maliva-v6/tools/occu_full.py 