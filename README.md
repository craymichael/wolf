<div align="center">
   <img src="./docs/images/wolf.png" width="600"><br><br>
</div>

-----------------------------------------------

**Wolf** is an open source library for Invertible Generative (Normalizing) Flows.

This is the code we used in the following papers

>[Decoupling Global and Local Representations via Invertible Generative Flows](https://vixra.org/abs/2004.0222)

>Xuezhe Ma, Xiang Kong, Shanghang Zhang and Eduard Hovy

>ICLR 2021

>[MaCow: Masked Convolutional Generative Flow](https://arxiv.org/abs/1902.04208)

>Xuezhe Ma, Xiang Kong, Shanghang Zhang and Eduard Hovy

>NeurIPS 2019

## Requirements
* Python >= 3.6
* Pytorch >= 1.3.1
* apex
* lmdb >= 0.94
* overrides 


## Installation
1. Install [NVIDIA-apex](https://github.com/NVIDIA/apex).
2. Install [Pytorch and torchvision](https://pytorch.org/get-started/locally/)

## Decoupling Global and Local Representations from/for Image Generation

### Switch Operation
<img src="./docs/images/switch.png" width="600"/>

### CelebA-HQ Samples
<img src="./docs/images/celeba_main.png" width="600"/>

### Running Experiments
First go to the experiments directory:
```base
cd experiments
```
Training a new CIFAR-10 model:
```base
python -u train.py \
    --config  configs/cifar10/glow-gaussian-uni.json \
    --epochs 15000 --valid_epochs 10
    --batch_size 512 --batch_steps 2 --eval_batch_size 1000 --init_batch_size 2048 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 32 --n_bits 8 \
    --data_path <data path> --model_path <model path>
```

#### Zach added
```shell
python distributed.py \
    --config configs/imagenet/64x64/glow/glow-base-uni.json \
    --epochs 15000 --valid_epochs 1 \
    --batch_size 256 --batch_steps 16 --eval_batch_size 500 --init_batch_size 2048 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 200 --weight_decay 5e-4 --grad_clip 0 \
    --image_size 64 --n_bits 8 --lr_decay 0.999997 \
    --dataset imagenet --train_k 3 \
    --data_path '/project01/cvrl/datasets/imagenet64/as_dirs/' --model_path 'models/imagenet/' \
    --nnodes 1 --nproc_per_node 4 --node_rank 0 --master_addr 127.0.0.1 --master_port 29500 | tee loggy-mcloggyface-"$(date -Is)".log


cifar10 144.8 s/epoch h (measured on four Tesla V100 GPUs)
Guesstimate: 1281167 / 50000 * 4 * 144.8 = 14841 s/epoch on ImageNet...
    This is about 4 hours per epoch, assuming the model is the same size
```

The hyper-parameters for other datasets are provided in the paper.
#### Note:
 - Config files, including refined version of Glow and MaCow, are provided [here](https://github.com/XuezheMax/wolf/tree/master/experiments/configs).
 - The argument --batch_steps is used for accumulated gradients to trade speed for memory. The size of each segment of data batch is batch-size / (num_gpus * batch_steps).
 - For distributed training on multi-GPUs, please use ```distributed.py``` or ```slurm.py```, and 
refer to the pytorch distributed parallel training [tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
 - Please check details of arguments [here](https://github.com/XuezheMax/wolf/blob/master/experiments/options.py).
 
## MaCow: Masked Convolutional Generative Flow
We also implement the MaCow model with distributed training supported. To train a new MaCow model, please use the MaCow config files for different datasets.

## References
```
@InProceedings{decoupling2021,
    title = {Decoupling Global and Local Representations via Invertible Generative Flows},
    author = {Ma, Xuezhe and Kong, Xiang and Zhang, Shanghang and Hovy, Eduard},
    booktitle = {Proceedings of the 9th International Conference on Learning Representations (ICLR-2021)},
    year = {2021},
    month = {May},
}

@incollection{macow2019,
    title = {MaCow: Masked Convolutional Generative Flow},
    author = {Ma, Xuezhe and Kong, Xiang and Zhang, Shanghang and Hovy, Eduard},
    booktitle = {Advances in Neural Information Processing Systems 33, (NeurIPS-2019)},
    year = {2019},
    publisher = {Curran Associates, Inc.}
}
```
