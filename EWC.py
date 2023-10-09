from copy import deepcopy
import torch
from torch.autograd import Variable
from torch import autograd
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

        # 计算 Fisher 信息矩阵
        self.fisher_info_gen, self.fisher_info_disc = self.compute_fisher()

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
        gen_lls = []
        disc_lls = []

        # 遍历数据加载器以计算 Fisher 信息
        for i, data in enumerate(self.dataloader):
            real = data[0].to(self.device)
            noise = torch.randn(real.size()[0], 100, 1, 1, device=self.device)
            fake = self.generator(noise)
            pred_real = self.discriminator(real)
            pred_fake = 1 - self.discriminator(fake)
            output = -torch.log(torch.cat((pred_real, pred_fake)))
            disc_lls.append(output)

        # 计算判别器的 Fisher 信息矩阵
        disc_lls = torch.cat(disc_lls).unbind()
        disc_ll_grads = zip(*[autograd.grad(l, self.discriminator.parameters(),retain_graph=(i < len(disc_lls))) for i, l in enumerate(disc_lls, 1)])

        for i in range(int(self.dataloader.dataset.targets.size()[0] / self.dataloader.batch_size)):
            noise = torch.randn(self.dataloader.batch_size, 100, 1, 1, device=self.device)
            fake = self.generator(noise)
            output = -torch.log(self.discriminator(fake))
            gen_lls.append(output)

        # 计算生成器的 Fisher 信息矩阵
        gen_lls = torch.cat(gen_lls).unbind()
        gen_ll_grads = zip(*[autograd.grad(l, self.generator.parameters(),retain_graph=(i < len(gen_lls))) for i, l in enumerate(gen_lls, 1)])
        gen_ll_grads = [torch.stack(gs) for gs in gen_ll_grads]
        gen_fisher_diagonals = [(g ** 2).mean(0) for g in gen_ll_grads]
        disc_ll_grads = [torch.stack(gs) for gs in disc_ll_grads]
        disc_fisher_diagonals = [(g ** 2).mean(0) for g in disc_ll_grads]

        # 获取生成器和判别器参数的名称，并返回 Fisher 信息矩阵
        gen_names = [n for n, p in self.generator.named_parameters()]
        disc_names = [n for n, p in self.discriminator.named_parameters()]
        return ({n: f.detach() for n, f in zip(gen_names, gen_fisher_diagonals)},
                {n: f.detach() for n, f in zip(disc_names, disc_fisher_diagonals)})

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
