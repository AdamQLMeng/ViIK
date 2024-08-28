# ViIK: Flow-based Vision Inverse Kinematics Solver with Fusing Collision Checking

Code for reproducing the experiments in the paper:
>Qinglong Meng, Chongkun Xia, and Xueqian Wang, "ViIK: Flow-based Vision Inverse Kinematics Solver with Fusing Collision Checking
"
>
>[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2408.11293-blue)](https://arxiv.org/abs/2408.11293)
## Setup
Setup environment:
```
bash env.sh
```

## Dataset

Dataset used in this paper can be downloaded at:
```
baidu disk: https://pan.baidu.com/s/1lYbM5LrV1xWQhFKZYYNRwQ password: 8888
```
also can create a new dataset by running:
```
python scripts/build_dataset.py --robot_name=panda --env_name_list=${YOUR-OWN-ENV}$ --obs_mesh_root=${MESH-FILES-ROOT}$
```
## Train
An example for training ViIK-2:
```
python train.py --robot_name=panda --num_nodes_value=24 --env_name_list=${2ENV4TRAINING}$ --gamma=0.988553095 --learning_rate=3e-5
```

## Evaluate
An example for evaluating ViIK-2:
```
python evaluate.py --robot_name=panda --num_nodes_value=24 --model_file=${PATH2CKPT}$
```
