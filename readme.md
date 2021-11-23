# SE(3)-eSCOPE

### [video]() | [paper](https://arxiv.org/pdf/2111.00190.pdf) | [website](https://dragonlong.github.io/equi-pose/)

##### [Leveraging SE(3) Equivariance for Self-Supervised Category-Level Object Pose Estimation](https://arxiv.org/pdf/2111.00190.pdf)


[Xiaolong Li](http://dragonlong.github.io/),     [Yijia Weng](https://halfsummer11.github.io/),     [Li Yi](https://cs.stanford.edu/~ericyi/) ,    [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/index.html),     [A. Lynn Abbott](https://ece.vt.edu/people/profile/abbott),     [Shuran Song](https://www.cs.columbia.edu/~shurans/),     [He Wang](https://hughw19.github.io/)

NeurIPS 2021

SE(3)-eSCOPE is a self-supervised learning framework to estimate category-level 6D object pose from single 3D point clouds, with **no ground-truth pose annotations**, **no GT CAD models**, and **no multi-view supervision** during training. The key to our method is to disentangle shape and pose through an invariant shape reconstruction module and an equivariant pose estimation module, empowered by SE(3) equivariant point cloud networks and reconstruction loss.

<img src="./imgs/0726.gif" width="1080">

### News

**\[2021-11\]** We release the training code for 5 categories.

### Prerequisites
The code is built and tested with following libraries:

- Python>=3.6
- [PyTorch](https://pytorch.org/get-started/locally/)/1.7.1
- gcc>=6.1.0
- cmake
- cuda/11.0.1, or cuda/11.1 for newer GPUs
- cudnn

#### Recommended Installation
```bash
# 1. install python environments
conda create --name equi-pose python=3.6
source activate equi-pose
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 2. compile extra CUDA libraries
bash build.sh
```

### Data Preparation
You could find the subset we use for ModelNet40 directly [[drive_link](https://drive.google.com/file/d/1pP7jWtTzwYmcBmy5aJDCOGSStR2_HKt_/view?usp=sharing)], and our rendered depth point clouds dataset [[drive_link](https://drive.google.com/file/d/1nfpocK1RVHnlubb29q2t3tiLZdlqG2qe/view?usp=sharing)], download and put them into your own 'data' folder. check [`global_info.py`](./global_info.py) for codes and data paths.
- `project_path` should contain the `equi-pose` code folder;
- `second_path` is set to store logs, checkpoints, results, and etc;
- check [`configs/dataset/modelnet40_complete.yaml`](./configs/dataset/modelnet40_complete.yaml) to set `dataset_path`;
- check [`configs/dataset/modelnet40_partial.yaml`](./configs/dataset/modelnet40_partial.yaml) to set `dataset_path`;

### Training
You may run the following code to train the model from scratch:

```bash
python main.py exp_num=[experiment_id] training=[name_training] datasets=[name_dataset] category=[name_category]
```

For example, to train the model on completet airplane, you may run

```bash
python main.py exp_num='1.0' training="complete_pcloud" dataset="modelnet40_complete" category='airplane' use_wandb=True
```

### Testing Pretrained Models
Some of our pretrained checkpoints have been released, check [[drive_link](https://drive.google.com/drive/folders/1i8EvIugHF8kmk-sgpzAhiQM2a4p1R7cu?usp=sharing)]. Put them in the '`second_path`/models' folder. You can run the following command to test the performance;
```bash
python main.py exp_num=[experiment_id] training=[name_training] datasets=[name_dataset] category=[name_category] eval=True save=True
```

For example, to test the model on complete airplane category or partial airplane, you may run

```bash
python main.py exp_num='0.813' training="complete_pcloud" dataset="modelnet40_complete" category='airplane'
eval=True save=True

```

```bash
python main.py exp_num='0.913r' training="partial_pcloud" dataset="modelnet40_partial" category='airplane' eval=True save=True
```
Note: add "use_fps_points=True" to get slightly better results; for your own datasets, add 'pre_compute_delta=True' and use example canonical shapes to compute pose misalignment first.

### Visualization
Check out my script [`demo.py`](./utils/demo.py) or [`teaser.py`](./utils/teaser.py) for some hints.
<img src="./imgs/0.8581_modelnet40aligned_chair.gif" width="1080">

## Citation

If you use this code for your research, please cite our paper.

```@inproceedings{
@inproceedings{li2021leveraging,
    title={Leveraging SE (3) Equivariance for Self-supervised Category-Level Object Pose Estimation from Point Clouds},
    author={Li, Xiaolong and Weng, Yijia and Yi, Li and Guibas, Leonidas and Abbott, A Lynn and Song, Shuran and Wang, He},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021}
  }

```
We thank Haiwei Chen for the helpful discussions on equivariant neural networks.
