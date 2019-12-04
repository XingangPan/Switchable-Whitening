## Switchable Whitening (SW)

### Paper

Xingang Pan, Xiaohang Zhan, Jianping Shi, Xiaoou Tang, Ping Luo. ["Switchable Whitening for Deep Representation Learning"](https://arxiv.org/abs/1904.09739), ICCV2019.  

### Introduction

* Switchable Whitening unifies various whitening and standardization techniques in a general form, and adaptively learns their importance ratios for different tasks.
* This repo is for ImageNet classification. We also provide the code for Syncronized SW at `models/ops/sync_switchwhiten.py`, which could be used for detection and segmentation.

### Requirements

* python>=3.6
* pytorch>=1.0.1
* others

    ```sh
    pip install -r requirements.txt
    ```

### Results

Top1/Top5 error on the ImageNet validation set are reported. The pretrained models with SW are available at [link](https://drive.google.com/open?id=1dUK0hJ_FYXP45LDJnAW_vjNGRoAP4Byb).

| Model                 | BN | SN | BW | SW (BW+IW) |
| -------------------   | ------------------ | ------------------ | ------------------ | ------------------ |
| ResNet-50             | 23.58/7.00  | 23.10/6.55  | 23.31/6.72  | 22.07/6.04  |
| ResNet-101            | 22.48/6.23  | 22.01/5.91  | 22.10/5.98  | 20.87/5.54  |
| DenseNet-121          | 24.96/7.85  | 24.38/7.26  | 24.56/7.55  | 23.56/6.85  |
| DenseNet-169          | 24.02/7.06  | 23.16/6.55  | 23.24/6.65  | 22.48/6.29  |

### Before Start
1. Clone the repository  
    ```Shell
    git clone https://github.com/XingangPan/Switchable-Whitening.git
    ```

2. Download [ImageNet](http://image-net.org/download-images) dataset. You may follow the instruction at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) to process the validation set.

### Training
1. Train with nn.DataParallel
    ```Shell
    sh experiments/resnet50_sw/run.sh  # remember to modify --data to your ImageNet path
    ```
2. Distributed training based on [slurm](https://slurm.schedmd.com/)
    ```Shell
    sh experiments/resnet50_sw/run_slurm.sh ${PARTITION}
    ```

### Practical concerns

* Inspired by [IterativeNorm](https://arxiv.org/abs/1904.03441), SW is accelarated via Newton's iteration.
* For SW, 4x64 (GPU number x batchsize) performs slightly better than 8x32.

### TODO
1. We would release implementation of Syncronized SW in instance segmentation later.

### Citing SW
```  
@inproceedings{pan2018switchable,
  author = {Pan, Xingang and Zhan, Xiaohang and Shi, Jianping and Tang, Xiaoou and Luo, Ping},
  title = {Switchable Whitening for Deep Representation Learning},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```  
