# InPL: Pseudo-labeling the Inliers First for Imbalanced Semi-supervised Learning

This is the implementation for [InPL: Pseudo-labeling the Inliers First for Imbalanced Semi-supervised Learning](https://openreview.net/forum?id=m6ahb1mpwwX) published at ICLR 2023. [[arXiv](https://arxiv.org/abs/2303.07269)]

## Requirements
This code is built largely on [TorchSSL](https://github.com/TorchSSL/TorchSSL) (thanks to the contribution of the authors). Thus, the requirement follows TorchSSL. This code is lastely tested with 

- python 3.9.16
- PyTorch 1.12.0

Installing the above PyTorch version following the official PyTorch instruction should satisfy most requirements. Other packages can be installed following runtime error message. 

## Usage 
Training on Cifar10-LT

`python inpl.py  --c config/inpl/inpl_cifar10_lt.yaml` 

Training on Cifar100-LT

`python inpl.py  --c config/inpl/inpl_cifar100_lt.yaml` 

## FAQ
1. **How to play with the codebase?**

There are two parts of InPL: energy-based thresholding and adaptive marginal loss (optional). The energy threshold and whether to use adaptive marginal loss can be modified in corresponding config files.

2. **How to select the optimal energy threshold for other datasets?**

This is tied to the limitation of energy score: it is difficult to interpret the value of energy scores. Thus, it requires some hyper-parameter tuning. This is also a limitation for using energy score for conventional OOD detection as [explained by the original authors](https://github.com/wetliu/energy_ood/issues/13).

3. **Explanation of using InPL in ABC**

ABC introduces an auxilliary balanced classifier in addition to the original classifier. When using InPL in ABC, we only apply the energy threshold on the auxilliary balanced classifier and leave the original classifier unchanged as we observe no explicit benefit of applying InPL on both. For results in the paper, the adaptive marginal loss can be applied on both classifiers. 

In terms of the energy threshold, as discussed in 2, the optimal threshold selected for FixMatch may not be optimal one for ABC. With our experiments, generally, a range from -5 to -7.5 can be a good initial starting point. 


4. Other questions

For other questions related to the paper, feel free to use github issues. I will regularly check it. 

## Acknowledgment

This code is built largely on [TorchSSL](https://github.com/TorchSSL/TorchSSL) (thanks to the contribution of the authors).
