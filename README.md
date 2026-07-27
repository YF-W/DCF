# [MedIA'26] [Breaking error coupling via divergent–convergent coordination for semi-supervised medical image segmentation](https://doi.org/10.1016/j.media.2026.104236)

This is the PyTorch implemention of our *Medical Image Analysis* 2026 paper "**Breaking error coupling via divergent–convergent coordination for semi-supervised medical image segmentation**" by [Yuxuan Wan](https://orcid.org/0009-0001-8874-0470), Zhixuan Chen, Yuquan Xu, Yiyang Xu, Mingfeng Li and [Yuefei Wang](https://orcid.org/0000-0003-3032-1852)

## 👀 Abstract
> This study addresses the challenges of Error Coupling and model homogenization that commonly arise in dual-model collaborative learning for semi-supervised medical image segmentation by proposing a Divergent-Convergent Framework (DCF). The core innovation of this framework lies in abandoning the traditional blind pursuit of strong prediction consistency and instead dynamically quantifying the confidence and disagreement between the two models through a Guidance Mask (GM). In regions of high confidence and low disagreement, a Convergence Stabilization Mechanism is applied to reinforce the learning of robust pseudo-labels; in regions of low confidence or high disagreement, a Divergent Exploration Mechanism is activated, guiding the models to perform differentiated exploration along two dimensions: internal semantic confusion and external feature orthogonality. This effectively maintains model diversity while suppressing the propagation of shared errors. Systematic experiments on six publicly available datasets—four 2D (ACDC, PROMISE12, Hippocampus, ATLAS) and two 3D (BraTS2019, Pancreas-CT)—demonstrate that DCF significantly outperforms state-of-the-art semi-supervised methods across multiple labeled ratios, including 5%, 10%, and 20%. Qualitative analyses and cross-domain evaluations further validate the method's advantages in segmenting regions with ambiguous boundaries and its robustness to domain shifts. Moreover, DCF can be directly applied to fine-tune segmentation foundation models such as MedSAM and MedSAM2, consistently improving their performance under the same labeled budget, thereby demonstrating its practical value as a label-efficient strategy for clinical deployment.

## 📂 Datasets

1. ACDC: [https://github.com/HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

2. PROMISE12: [https://promise12.grand-challenge.org/](https://promise12.grand-challenge.org/)

3. Hippocampus: [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

4. ATLAS: [https://atlas-challenge.u-bourgogne.fr/dataset](https://atlas-challenge.u-bourgogne.fr/dataset)

5. BraTS2019: [https://github.com/HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

6. Pancreas-CT: [https://github.com/ycwu1997/MC-Net](https://github.com/ycwu1997/MC-Net)

**Note**: The data format and organization follow the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) convention. If you encounter any difficulties, please feel free to contact us.

## ⚡ Usage
### 1. Clone the repository

```bash
git clone https://github.com/YF-W/DCF.git
cd DCF/code
```
### 2. Start Training
#### 2D Setting
```bash
python train_2D.py --root_path ../data/ACDC --exp DCF --label_ratio 10%
```
#### 3D Setting:
```bash
python train_3D_pre_train.py --dataset_name BraTS2019 --exp DCF --label_ratio 10% --model voxresnet
python train_3D_pre_train.py --dataset_name BraTS2019 --exp DCF --label_ratio 10% --model attention_unet
python train_3D.py --dataset_name BraTS2019 --exp DCF --label_ratio 10%
```

### 3. Start Testing
#### 2D Setting
```bash
python test_2D.py --root_path ../data/ACDC --exp DCF --label_ratio 10%
```
#### 3D Setting:
```bash
python test_3D.py --dataset_name BraTS2019 --exp DCF --label_ratio 10%
```
## 📜 Citation

If you find this project useful, please consider citing:

```bibtex
@article{DCF,
  title={Breaking error coupling via divergent–convergent coordination for semi-supervised medical image segmentation},
  author={Yuxuan Wan and Zhixuan Chen and Yuquan Xu and Yiyang Xu and Mingfeng Li and Yuefei Wang},
  journal={Medical Image Analysis},
  year={2024},
  pages = {104236},
  year = {2026},
  doi = {https://doi.org/10.1016/j.media.2026.104236}
}
```

## 🙏 Acknowledgements
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.