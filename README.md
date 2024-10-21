# CuNeRF
The source code for our paper "**[CuNeRF: Cube-Based Neural Radiance Field for Zero-Shot Medical Image Arbitrary-Scale Super Resolution](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_CuNeRF_Cube-Based_Neural_Radiance_Field_for_Zero-Shot_Medical_Image_Arbitrary-Scale_ICCV_2023_paper.pdf)**", [Zixuan Chen](https://narcissusex.github.io), [Lingxiao Yang](https://zjjconan.github.io/), [Jian-Huang Lai](https://cse.sysu.edu.cn/content/2498), [Xiaohua Xie](https://cse.sysu.edu.cn/content/2478), *IEEE/CVF International Conference on Computer Vision* (**ICCV**), 2023.

<p align="center">
  <a href="https://narcissusex.github.io/CuNeRF/">Project Page</a> |
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_CuNeRF_Cube-Based_Neural_Radiance_Field_for_Zero-Shot_Medical_Image_Arbitrary-Scale_ICCV_2023_paper.pdf">Paper</a> 
</p>
<div align=center>
<img width="1148" alt="framework" src="assets/cunerf.png">
</div>

## Download dataset

```bash
MSD: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
KiTS19: https://github.com/neheller/kits19
```


## 1) Get start

* Python 3.9.x
* CUDA 11.1 or *higher*
* NVIDIA RTX 3090
* Torch 1.8.0 or *higher*

**Create a python env using conda**
```bash
conda create -n cunerf python=3.9 -y
```

**Install the required libraries**
```bash
bash setup.sh
```

**[option] Install FFmpeg**
```bash
apt install ffmpeg -y
```


## 2) Training CuNeRF for medical volumes
```bash
python run.py <expname> --cfg <config file> --scale <SR scale> --mode train --file <filepath>

python run.py msd_brain_tumour --cfg configs/example.yaml --modality T1w --scale 2 --mode train --file dataset/MSD/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --save_map --resume
```
See *example_train.sh* for details, we also provide an example config file in the *configs* dir.

## 3) Arbitrary rendering for medical slices
Render slices at arbitrary positions (*zpos*: $-0.1$ ~ $0.1$), scales ($1$.x ~ $2$.x) and viewpoints (*angles*: $0$ ~ $360$ degrees) with an rotation axis $[1,1,0]$:
```bash
python run.py <expname> --cfg <config file> --mode test --file <filepath> --scales 1 2 --zpos -0.1 0.1 --angles 0 360 --axis 1 1 0 --asteps 45 

python run.py msd_brain_tumour --cfg configs/example.yaml --mode test --file dataset/MSD/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --resume_type psnr --axis 1 1 0 --asteps 45 --save_map --is_details --is_gif --is_video --modality T1w --scales 1.5 --zpos -0.05 --angles 45
```
See *example_test.sh* for details.

## Citation

```tex
@InProceedings{Chen_2023_ICCV,
author    = {Chen, Zixuan and Yang, Lingxiao and Lai, Jian-Huang and Xie, Xiaohua},
title     = {CuNeRF: Cube-Based Neural Radiance Field for Zero-Shot Medical Image Arbitrary-Scale Super Resolution},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month     = {October},
year      = {2023},
pages     = {21185-21195}
}
```

## Acknowledgement 

We build our project based on **[NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch)**. We thank them for their wonderful work and code release.