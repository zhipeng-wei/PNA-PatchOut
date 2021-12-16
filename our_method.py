import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
import copy
from utils import ROOT_PATH
from functools import partial
import copy
import pickle as pkl
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import params
from model import get_model

class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:,None, None]).add_(mean[:,None,None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.sub_(mean[:,None,None]).div_(std[:,None,None])
        return inps

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i,filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1,2,0]) # c,h,w to h,w,c
            image[image<0] = 0
            image[image>1] = 1
            image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
            # print ('Saving to ', save_path)
            image.save(save_path)

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

class OurAlgorithm(BaseAttack):
    def __init__(self, model_name, ablation_study='1,1,1', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False):
        super(OurAlgorithm, self).__init__('OurAlgorithm', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps

        self.ablation_study = ablation_study.split(',')
        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        if self.ablation_study[2] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')
    
    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            if self.ablation_study[0] == '1':
                print ('Using Pathes')
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            else:
                print ('Not Using Pathes')
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))

            if self.ablation_study[1] == '1':
                print ('Using L2')
                cost1 = self.loss_flag * loss(outputs, labels).cuda()
                cost2 = torch.norm(perts)
                cost = cost1 + self.lamb * cost2
            else:
                print ('Not Using L2')
                cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None

class OurAlgorithm_MI(BaseAttack):
    def __init__(self, model_name, ablation_study='1,1,1', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(OurAlgorithm_MI, self).__init__('OurAlgorithm_MI', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.decay = decay

        self.ablation_study = ablation_study.split(',')
        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        if self.ablation_study[2] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')
    
    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            if self.ablation_study[0] == '1':
                print ('Using Pathes')
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            else:
                print ('Not Using Pathes')
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))

            if self.ablation_study[1] == '1':
                print ('Using L2')
                cost1 = self.loss_flag * loss(outputs, labels).cuda()
                cost2 = torch.norm(perts)
                cost = cost1 + self.lamb * cost2
            else:
                print ('Not Using L2')
                cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None

class OurAlgorithm_SGM(BaseAttack):
    def __init__(self, model_name, sgm_control='1,0', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False):
        super(OurAlgorithm_SGM, self).__init__('OurAlgorithm_SGM', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.sgm_control = sgm_control.split(',')

        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        self._register_model()

    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        def mlp_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])
        
        def attn_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
        mlp_hook_func = partial(mlp_mask_grad, gamma=0.5)
        attn_hook_func = partial(attn_mask_grad, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.blocks[i].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.blocks[i].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks_token_only[block_ind-24].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage2[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.stage2[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':           
                        self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(attn_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()

        for i in range(self.steps):
            perts.requires_grad_()
            add_perturbation = self._generate_samples_for_interactions(perts, i)
            outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            cost1 = self.loss_flag * loss(outputs, labels).cuda()
            cost2 = torch.norm(perts)

            cost = cost1 + self.lamb * cost2
            cost.backward()
            grad = perts.grad.data
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None

class OurAlgorithm_SGM_MI(BaseAttack):
    def __init__(self, model_name, sgm_control='1,0', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(OurAlgorithm_SGM_MI, self).__init__('OurAlgorithm_SGM_MI', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.sgm_control = sgm_control.split(',')
        self.decay = decay

        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        self._register_model()

    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        def mlp_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])
        
        def attn_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
        mlp_hook_func = partial(mlp_mask_grad, gamma=0.5)
        attn_hook_func = partial(attn_mask_grad, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.blocks[i].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.blocks[i].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks_token_only[block_ind-24].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage2[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.stage2[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':           
                        self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(attn_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()

        for i in range(self.steps):
            perts.requires_grad_()
            add_perturbation = self._generate_samples_for_interactions(perts, i)
            outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            cost1 = self.loss_flag * loss(outputs, labels).cuda()
            cost2 = torch.norm(perts)

            cost = cost1 + self.lamb * cost2
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None