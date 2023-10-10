from copy import deepcopy
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

class EWC(object):
    # 初始化 EWC 类的构造函数
    def __init__(self, generator: nn.Module, discriminator: nn.Module, device, dataloader):
        # 存储生成器和判别器模型
        self.generator = generator
        self.discriminator = discriminator

        # 检测可用的计算设备（GPU 或 CPU）
        self.device = device

        # 创建数据加载器用于加载数据批次
        self.dataloader = dataloader

        # 提取生成器和判别器的可训练参数
        self.gen_params = {n: p for n, p in self.generator.named_parameters() if p.requires_grad}
        self.disc_params = {n: p for n, p in self.discriminator.named_parameters() if p.requires_grad}

        # 计算 Fisher 信息矩阵，即参数的重要性权重
        self.fisher_info_disc, self.fisher_info_gen = self.compute_fisher()

        # 创建用于存储模型参数的副本
        self.gen_star_vars= {}
        self.disc_star_vars = {}
        for n, p in deepcopy(self.gen_params).items():
            if torch.cuda.is_available():
                p = p.cuda()
            self.gen_star_vars[n] = Variable(p.data)
        for n, p in deepcopy(self.disc_params).items():
            if torch.cuda.is_available():
                p = p.cuda()
            self.disc_star_vars[n] = Variable(p.data)

    # 计算 Fisher 信息矩阵的函数
    def compute_fisher(self):
        # 定义判别器和生成器的Fisher 信息矩阵
        disc_precision_matrices = {}
        gen_precision_matrices = {}
        # 初始化
        for n, p in deepcopy(self.disc_params).items():
            p.data.zero_()
            disc_precision_matrices[n] = Variable(p.data)
        for n, p in deepcopy(self.gen_params).items():
            p.data.zero_()
            gen_precision_matrices[n] = Variable(p.data)

        temp_len = len(self.dataloader)

        # 计算判别器的 Fisher 信息矩阵
        self.discriminator.eval()
        for i, data in enumerate(self.dataloader):
            real = data[0].to(self.device)
            noise = torch.randn(real.size()[0], 100, 1, 1, device=self.device)
            fake = self.generator(noise)
            pred_real = self.discriminator(real)
            pred_fake = self.discriminator(fake)
            self.discriminator.zero_grad()
            loss_real = torch.mean(pred_real)
            loss_fake = torch.mean(pred_fake)
            loss_disc = -loss_real + loss_fake
            loss_disc.backward()
            for n, p in self.discriminator.named_parameters():
                disc_precision_matrices[n].data += p.grad.data ** 2 / temp_len
        disc_precision_matrices = {n: p for n, p in disc_precision_matrices.items()}

        # 计算生成器的 Fisher 信息矩阵
        self.generator.eval()
        for i in range(len(self.dataloader)):
            noise = torch.randn(self.dataloader.batch_size, 100, 1, 1, device=self.device)
            fake = self.generator(noise)
            output = self.discriminator(fake)
            loss_gen = - torch.mean(output)
            self.generator.zero_grad()
            loss_gen.backward()

            for n, p in self.generator.named_parameters():
                gen_precision_matrices[n].data += p.grad.data ** 2 / temp_len
        gen_precision_matrices = {n: p for n, p in gen_precision_matrices.items()}

        return disc_precision_matrices, gen_precision_matrices

    # 计算 EWC 损失的函数
    def penalty(self, model, gen=True):
        # 获取当前模型参数
        params = model.named_parameters()
        # 获取初始模型参数
        if gen:
            star_vars = self.gen_star_vars
            fisher = self.fisher_info_gen
        else:
            star_vars = self.disc_star_vars
            fisher = self.fisher_info_disc

        loss = 0
        for n, p in params:
            penalty = fisher[n] * (p - star_vars[n]) ** 2
            loss += penalty.sum()
        return loss
