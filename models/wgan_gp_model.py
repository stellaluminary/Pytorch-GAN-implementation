from .base_model import BaseModel
import models.networks as networks
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

logger = logging.getLogger('base')

class WGANGPModel(BaseModel):
    def __init__(self, opt):
        super(WGANGPModel, self).__init__(opt)

        # define fixed noise
        self.fixed_noise = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)
        self.batch_size = self.opt['Data_Param']['batch_size']

        # define self.model_names for saving pth file
        if self.is_train == 'train':
            self.model_names = ['G', 'D']
            self.visual_names = ['data', 'fake', 'fake_fixed']
        else:
            self.model_names = ['G']
            self.visual_names = ['fake']

        # define self.loss_names for saving loss log file
        self.loss_names = ['G', 'D']

        # define generator and discriminator
        self.netG = networks.define_G(opt, 'G').to(self.device)

        if self.is_train == 'train':
            self.netD = networks.define_D(opt, 'D').to(self.device)
            self.netG.train()
            self.netD.train()

            # define optimizers : D & G
            self.optim_G = optim.Adam(params=self.netG.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optim_D = optim.Adam(params=self.netD.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))

            self.optimizers.append(self.optim_G)
            self.optimizers.append(self.optim_D)

    def feed_data(self, data):
        self.data = data['A'].to(self.device)
        self.data_paths = data['A_paths']
        self.noise = data['noise'].to(self.device)

    def forward(self):

        self.fake = self.netG(self.noise)  # x: noise -> G(x): netG(x) = fake
        self.fake_fixed = self.netG(self.fixed_noise)

    def optimize_parameters(self, idx):

        # ------ define fake data ------

        self.forward()

        # -------------------------- train generator G --------------------------

        if idx % self.opt['Train']['n_critic'] == 0:
            self.set_requires_grad([self.netD], requires_grad=False)

            disc_fake = self.netD(self.fake)
            self.loss_G = -torch.mean(disc_fake)

            self.optim_G.zero_grad()
            self.loss_G.backward()
            self.optim_G.step()

        # -------------------------- train discriminator D --------------------------

        self.set_requires_grad([self.netD], requires_grad=True)

        disc_real = self.netD(self.data)
        disc_fake = self.netD(self.fake.detach())

        gp = self.gradient_penalty(self.data, self.fake.detach())

        self.loss_D = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.opt['Train']['lambda'] * gp

        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

    def test(self):

        self.fixed_noise2 = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)
        with torch.no_grad():
            self.fake = self.netG(self.fixed_noise2)

    def gradient_penalty(self, real_samples, fake_samples):
        BATCH_SIZE, C, H, W = real_samples.shape
        # uniform distribution alpha
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated_images = real_samples * alpha + fake_samples * (1 - alpha)

        d_interpolate = self.netD(interpolated_images.requires_grad_(True))

        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=d_interpolate,
            grad_outputs=torch.ones_like(d_interpolate),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
