from .base_model import BaseModel
import models.networks as networks
import logging
import torch
import torch.optim as optim
import torch.nn as nn

logger = logging.getLogger('base')

class DCGANModel(BaseModel):

    def __init__(self, opt):
        super(DCGANModel, self).__init__(opt)

        # define fixed noise
        self.fixed_noise = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)

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

            # define loss
            self.cri_gan = nn.BCEWithLogitsLoss().to(self.device)

            # define optimizers : D & G
            self.optim_G = optim.Adam(params=self.netG.parameters(),
                                      lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optim_D = optim.Adam(params=self.netD.parameters(),
                                      lr=opt['Train']['lr'],
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

        self.set_requires_grad([self.netD], requires_grad=False)

        disc_fake = self.netD(self.fake)
        self.loss_G = self.cri_gan(disc_fake, torch.ones_like(disc_fake).to(self.device))

        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()

        # -------------------------- train discriminator D --------------------------

        self.set_requires_grad([self.netD], requires_grad=True)

        disc_real = self.netD(self.data)
        loss_D_real = self.cri_gan(disc_real, torch.ones_like(disc_real).to(self.device))

        disc_fake = self.netD(self.fake.detach())
        loss_D_fake = self.cri_gan(disc_fake, torch.zeros_like(disc_fake).to(self.device))

        self.loss_D = (loss_D_real + loss_D_fake)

        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

    def test(self):
        # define fixed noise2
        self.fixed_noise2 = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)

        with torch.no_grad():
            self.fake = self.netG(self.fixed_noise2)