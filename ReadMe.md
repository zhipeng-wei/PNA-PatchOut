<div align="center">

<h1><a href="https://ojs.aaai.org/index.php/AAAI/article/download/20169/19928">Towards Transferable Adversarial Attacks on Vision Transformers</a></h1>

**AAAI 2022**

<h1><a href="https://ieeexplore.ieee.org/abstract/document/10319323">Towards transferable adversarial attacks on image and video transformers</a></h1>

**IEEE Transactions on Image Processing ( Volume: 32)**

</div>

If you use our method for attacks in your research, please consider citing
```
@inproceedings{wei2022towards,
  title={Towards transferable adversarial attacks on vision transformers},
  author={Wei, Zhipeng and Chen, Jingjing and Goldblum, Micah and Wu, Zuxuan and Goldstein, Tom and Jiang, Yu-Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={2668--2676},
  year={2022}
}
@article{wei2023towards,
  title={Towards transferable adversarial attacks on image and video transformers},
  author={Wei, Zhipeng and Chen, Jingjing and Goldblum, Micah and Wu, Zuxuan and Goldstein, Tom and Jiang, Yu-Gang and Davis, Larry S},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={6346--6358},
  year={2023},
  publisher={IEEE}
}
```
# Introduction
To do.


# Environment
Recover the environment by
```
conda env create -f environment_transformer.yml
```

# Attacked Dataset
The used datasets are sampled from ImageNet. Unzip clean_resized_images.zip to **ROOT_PATH** of utils.py.

# Models
ViTs models from [timm](https://github.com/rwightman/pytorch-image-models): 
* vit_base_patch16_224
* deit_base_distilled_patch16_224
* levit_256
* pit_b_224
* cait_s24_224
* convit_base
* tnt_s_patch16_224
* visformer_small   

CNNs and robustly trained CNNs from [TI](https://github.com/dongyp13/Translation-Invariant-Attacks) and [here](https://github.com/tensorflow/models/tree/benchmark/research/adv_imagenet_models).

# Implementation
Change **ROOT_PATH** of utils.py.
### attack
```
python our_attack.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name vit_base_patch16_224 --filename_prefix yours 
```
* attack: the attack method, OurAlgorithm, OurAlgorithm_MI or OurAlgorithm_SGM
* model_name: white-box model name, vit_base_patch16_224, pit_b_224, cait_s24_224, visformer_small
* filename_prefix: additional names for the output file

### evaluate
```
sh run_evaluate.sh gpu model_{model_name}=method_{attack}-{filename_prefix}
```

