# AAAI-2022 Paper
Towards Transferable Adversarial Attacks on Vision Transformers

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

# Citation
If you use our method for attacks in your research, please consider citing
```
@article{Wei2021TowardsTA,
  title={Towards Transferable Adversarial Attacks on Vision Transformers},
  author={Zhipeng Wei and Jingjing Chen and Micah Goldblum and Zuxuan Wu and Tom Goldstein and Yu-Gang Jiang},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.04176}
}
```
