
 <h3 align="center"><strong>Density-guided Translator Boosts Synthetic-to-Real Unsupervised Domain
Adaptive Segmentation of 3D Point Clouds</strong></h3>
 <p align="center">
      Zhimin Yuan, Wankang Zeng, Yanfei Su, Weiquan Liu, Ming Cheng, Yulan Guo, Cheng Wang
    <br>
    <a href=https://asc.xmu.edu.cn/ target='_blank'>ASC</a>,&nbsp;Xiamen University
  </p>

This is a official code release of [DGT-ST](https://arxiv.org/pdf/2403.18469.pdf). This code is mainly based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

I was trying to clean the code to make it more readable. However, the cleaned code of **PCAN** can not reproduce the results shown in our paper. It may caused by the **instability** of the GAN training. Now, we are working on a new direction to obtain a warm-up model that is more stable than the GAN-based methods. So, we are not going to make too much effort for this. The code in this repo is only slightly cleaned but still nasty. 

The code was submitted in a hurry. Please contact me if you have any questions.

## LiDAR_UDA

We published a new [repo](https://github.com/yuan-zm/LiDAR_UDA) focusing on **self-training-based methods** for 3D outdoor driving scenario LiDAR point clouds UDA segmentation.

## Getting Started
```Shell
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda

# pytorch
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# MinkowskiEngine==0.5.4
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"


```
pip install other packages if needed.

Our released implementation is tested on
+ Ubuntu 18.04
+ Python 3.8 
+ PyTorch 1.10.1
+ MinkowskiEngine 0.5.4
+ NVIDIA CUDA 11.1
+ 3090 GPU

## DGT preparing
Please refer to [DGT.md](DGT.md) for the details.

## Train
1. Please train a SourceOnly model (see `SourceOnly_DGTST`) or directly download the [pretrained model](#model_zoo) and organize the downloaded files as follows
```
DGT_ST
├── preTraModel
│   ├── Syn2SK
│   │   │── SourceOnly
│   │   │── stage_1_PCAN
│   │   │── stage_2_SAC_LM
│   ├── Syn2Sp
│   │   │── SourceOnly
│   │   │── stage_1_PCAN
│   │   │── stage_2_SAC_LM
├── change_data
├── configs
```

#### Important notes: Please choose `PRETRAIN_PATH` carefully in the `config file`.

SynLiDAR -> SemanticKITTI:
### Stage 1: PCAN 
We use `checkpoint_val_Sp.tar` as the pre-trained model.

Note: The GAN training is unstable. If you can not reproduce the results in our paper, please **re-train** this model.

``` python train_DGT_ST.py --cfg=configs/SynLiDAR2SemanticKITTI/stage_1_PCAN.yaml```
### Stage 2: SAC-LM
We use `checkpoint_val_target_Sp.tar`, the model pretrained by PCAN, as the pre-trained model.

``` python train_DGT_ST.py --cfg=configs/SynLiDAR2SemanticKITTI/stage_2_SAC_LM.yaml```

### Baseline [LaserMix](https://arxiv.org/abs/2207.00026) for UDA
In order to be consistent with [CoSMix](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930575.pdf), we use `checkpoint_epoch_10.tar` as the pre-trained model.

``` python train_DGT_ST.py --cfg=configs/SynLiDAR2SemanticKITTI/LaserMix.yaml```

### Baseline [ADVENT](https://arxiv.org/abs/1811.12833)

```
python train_DGT_ST.py --cfg=configs/SynLiDAR2SemanticKITTI/ADVENT.yaml
```

## Test

For example, test the `semanticKITTI`.

<details><summary>Code</summary>

```
python infer.py \
        --checkpoint_path upload_model/syn2sk/stage_1_PCAN/checkpoint_val_target_Sp.tar \
        --result_dir res_pred/syn2sk/stage_1_PCAN \
        --batch_size 12 \
        --num_classes 20 \
        --dataset_name SemanticKITTI \
        --cfg configs/SynLiDAR2SemanticKITTI/stage_1_PCAN.yaml

python eval_performance.py \
        --dataset ~/dataset/semanticKITTI/dataset/sequences \
        --predictions res_pred/syn2sk/stage_1_PCAN \
        --sequences 08 \
        --num-classes 20 \
        --datacfg utils/semantic-kitti.yaml
```
</details>

test the `semanticPOSS`:
<details><summary>Code</summary>

```
python infer.py \
        --checkpoint_path upload_model/syn2sp/stage_1_PCAN/checkpoint_val_target_Sp.tar \
        --result_dir res_pred/syn2sp/stage_1_PCAN \
        --batch_size 12 \
        --num_classes 14 \
        --dataset_name SemanticPOSS \
        --cfg configs/SynLiDAR2SemanticPOSS/stage_1_PCAN.yaml

python eval_performance.py \
        --dataset ~/dataset/semanticPOSS/dataset/sequences \
        --predictions res_pred/syn2sp/stage_1_PCAN \
        --sequences 03 \
        --num-classes 14 \
        --datacfg utils/semantic-poss.yaml
```
</details>


## Model Zoo <a id="model_zoo"></a>


We release the checkpoints of SynLiDAR -> SemanticKITTI and SynLiDAR -> SemanticPOSS. You can directly use the provided model for testing. Then, you will get the same results as in Tab.1 and Tab. 2 in our [paper](https://arxiv.org/pdf/2403.18469.pdf).

#### Important notes: The provided checkpoints are all trained by our **unclean** code.

[Baidu (提取码：btod)](https://pan.baidu.com/s/1yPuNvFnDPnd9lBF-7I6UHw?pwd=btod) 

[Google](https://drive.google.com/drive/folders/1EuxDphixI579hBgiOQFoVz5KlVwHllkP?usp=sharing)


## Acknowledgement
Thanks for the following works for their awesome codebase.

[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

[SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)

[CoSMix](https://github.com/saltoricristiano/cosmix-uda)

[LaserMix](https://github.com/ldkong1205/LaserMix)

Although LaserMix is proposed for semi-supervised segmentation tasks, I found it very effective in 3D UDA segmentation tasks. The experimental results show that it is very stable during the training stage. We speculate that the spatial prior helps bridge the domain gap in UDA task. 

[LiDAR Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990175.pdf)

We use the method proposed by LiDAR Distillation [(code)](https://github.com/weiyithu/LiDAR-Distillation) to obtain the beam-label. 


## Citation

```
@inproceedings{DGTST,
    title={Density-guided Translator Boosts Synthetic-to-Real Unsupervised Domain Adaptive Segmentation of 3D Point Clouds},
    author={Zhimin Yuan, Wankang Zeng, Yanfei Su, Weiquan Liu, Ming Cheng, Yulan Guo, Cheng Wang},
    booktitle={CVPR},
    year={2024}
}
```