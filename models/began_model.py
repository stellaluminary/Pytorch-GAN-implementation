from .base_model import BaseModel
import models.networks as networks
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

logger = logging.getLogger('base')

class BEGANModel(BaseModel):
    def __init__(self, opt):
        super(BEGANModel, self).__init__(opt)

        # define fixed noise
        self.fixed_noise = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)
        self.batch_size = self.opt['Data_Param']['batch_size']

        self.lr = self.opt['Train']['lr']
        self.lr_decay_iter = self.opt['Train']['lr_decay_iter']

        # define k & gamma for equilibrium (Critical for BEGAN)
        self.k = self.opt['Train']['k']
        self.lambda_k = self.opt['Train']['lambda_k']
        self.gamma = self.opt['Train']['gamma']

        # define self.model_names for saving pth file
        if self.is_train == 'train':
            self.model_names = ['G', 'D']
            self.visual_names = ['data', 'fake', 'fake_fixed']
            self.add_value_names = ['k', 'B', 'M', 'lr']
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

            self.dataset_size = self.opt['dataset_size']

            # define optimizers : D & G
            self.optim_G = optim.Adam(params=self.netG.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optim_D = optim.Adam(params=self.netD.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optimizers.append(self.optim_G)
            self.optimizers.append(self.optim_D)

            # # define learning rate decay scheduler
            # self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optim_G, lr_lambda=lambda epoch: 0.95 ** epoch)
            # self.scheduler_D = optim.lr_scheduler.LambdaLR(self.optim_D, lr_lambda=lambda epoch: 0.95 ** epoch)
            # self.schedulers.append(self.scheduler_G)
            # self.schedulers.append(self.scheduler_D)

    def feed_data(self, data):
        self.data = data['A'].to(self.device)
        self.data_paths = data['A_paths']
        self.noise = data['noise'].to(self.device)

    def forward(self):
        self.fake = self.netG(self.noise)  # x: noise -> G(x): netG(x) = fake
        self.fake_fixed = self.netG(self.fixed_noise)

    def optimize_parameters(self, total_idx):

        # ------ define fake data ------

        self.forward()

        # -------------------------- train generator G --------------------------

        self.set_requires_grad([self.netD], requires_grad=False)

        disc_fake = self.netD(self.fake)
        self.loss_G = torch.mean(torch.abs(disc_fake - self.fake))

        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()

        # -------------------------- train discriminator D --------------------------

        self.set_requires_grad([self.netD], requires_grad=True)

        disc_real = self.netD(self.data)
        disc_fake = self.netD(self.fake.detach())

        loss_D_real = torch.mean(torch.abs(disc_real - self.data))
        loss_D_fake = torch.mean(torch.abs(disc_fake - self.fake.detach()))

        self.loss_D = loss_D_real - self.k * loss_D_fake

        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

        # -------------------------- Update k for objective --------------------------

        balance = (self.gamma * loss_D_real - loss_D_fake).item()
        self.k = self.k + self.lambda_k * balance
        self.k = min(max(self.k, 0), 1)

        M = loss_D_real + np.abs(balance)

        # define k, B, M for saving the log file.
        self.k = self.k
        self.B = balance
        self.M = M.item()

        if total_idx % self.lr_decay_iter == 0:
           self.adjust_learning_rate(total_idx)

    def test(self):
        self.fixed_noise2 = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)
        with torch.no_grad():
            self.fake = self.netG(self.fixed_noise2)

    def adjust_learning_rate(self, niter):
        self.lr = max(0.00004, self.lr * (0.95 ** (niter // self.lr_decay_iter)))
        for i in range(len(self.optimizers)):
            for param_group in self.optimizers[i].param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = self.lr
        print('[%d iterations learning rate] %.7f -> %.7f' % (niter, old_lr, self.lr))
