from .base_model import BaseModel
import models.networks as networks
import logging
import torch
import torch.optim as optim
import itertools
import torch.nn as nn
from utils.image_pool import ImagePool

logger = logging.getLogger('base')

class CycleGANModel(BaseModel):

    def __init__(self, opt):
        super(CycleGANModel, self).__init__(opt)

        self.lambdaA = opt['Train']['lambdaA']
        self.lambdaB = opt['Train']['lambdaB']
        self.lambda_idt = opt['Train']['lambda_identity']

        # define self.model_names for saving pth file
        if self.is_train == 'train':
            self.model_names = ['G_AtoB', 'G_BtoA', 'D_A', 'D_B']
        else:
            self.model_names = ['G_AtoB', 'G_BtoA']

        # define self.loss_names for saving loss log file
        self.loss_names = ['G_AtoB', 'G_BtoA', 'cycle_AtoB', 'cycle_BtoA', 'D_A', 'D_B', 'G', 'D']
        if self.lambda_idt:
            self.loss_names += ['idt_A', 'idt_B']

        # define self.model_names for saving img file
        self.visual_names = ['data_A', 'fake_B', 'rec_A', 'data_B', 'fake_A', 'rec_B']

        # define 2 generator and 2 discriminator
        self.netG_AtoB = networks.define_G(opt, 'G_AtoB').to(self.device)
        self.netG_BtoA = networks.define_G(opt, 'G_BtoA').to(self.device)

        if self.is_train == 'train':
            self.netD_A = networks.define_D(opt, 'D_A').to(self.device)
            self.netD_B = networks.define_D(opt, 'D_B').to(self.device)
            self.netG_AtoB.train()
            self.netG_BtoA.train()
            self.netD_A.train()
            self.netD_B.train()

            # define ImagePool for mixing data
            self.fake_A_pool = ImagePool(opt['Train']['pool_size'])
            self.fake_B_pool = ImagePool(opt['Train']['pool_size'])

            # define loss
            """            
            gan loss = lsgan's loss = MSE loss
            cycle consistency loss = l1 loss
            """
            self.cri_cycle = nn.L1Loss().to(self.device)
            self.cri_gan = nn.MSELoss().to(self.device)
            if self.lambda_idt:
                self.cri_idt = nn.L1Loss().to(self.device)

            # define optimizers : D & G
            """
            Choose optimizer params methods :
            1) itertools.chain(netG_A.parameters(), netG_B.parameters())
            2) list(netG_A.parameters(), netG_B.parameters())
            """
            self.optim_G = optim.Adam(params=itertools.chain(self.netG_AtoB.parameters(), self.netG_BtoA.parameters()),
                                      lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optim_D = optim.Adam(params=itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                      lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))

            self.optimizers.append(self.optim_G)
            self.optimizers.append(self.optim_D)

            # define learning rate dacy scheduler
            self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optim_G, lr_lambda=self.lr_scheduler_lambda)
            self.scheduler_D = optim.lr_scheduler.LambdaLR(self.optim_D, lr_lambda=self.lr_scheduler_lambda)

            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)

    def feed_data(self, data):
        self.data_A = data['A'].to(self.device)
        self.data_B = data['B'].to(self.device)

        self.data_A_paths = data['A_paths']
        self.data_B_paths = data['B_paths']

    def forward(self):
        # -------------- define fake_A, fake_B, reconstruction_A, reconstruction_B data --------------
        self.fake_B = self.netG_AtoB(self.data_A)  # x: data_A -> G(x): netG_AtoB(x) = fake_B
        self.rec_A = self.netG_BtoA(self.fake_B)  # G(x): fake_B -> F(G(x)): netG_BtoA(G(x)) = rec_A ~ x
        self.fake_A = self.netG_BtoA(self.data_B)  # y: data_B -> F(y): netG_BtoA(y) = fake_A
        self.rec_B = self.netG_AtoB(self.fake_A)  # F(y): fake_A -> G(F(y)): netG_AtoB(F(y)) = rec_B ~ y

    def optimize_parameters(self, idx):

        # ------ define fake_A, fake_B, reconstruction_A, reconstruction_B data ------
        self.forward()
        # -------------------------- train generator G --------------------------

        # A --> B
        self.set_requires_grad([self.netD_A, self.netD_B], requires_grad=False)
        disc_fake_B = self.netD_B(self.fake_B)
        self.loss_G_AtoB = self.cri_gan(disc_fake_B, torch.ones_like(disc_fake_B).to(self.device))

        # forward cycle consistent loss
        self.loss_cycle_AtoB = self.cri_cycle(self.rec_A, self.data_A)

        # B --> A
        disc_fake_A = self.netD_A(self.fake_A)
        self.loss_G_BtoA = self.cri_gan(disc_fake_A, torch.ones_like(disc_fake_A).to(self.device))

        # backward cycle consistent loss
        self.loss_cycle_BtoA = self.cri_cycle(self.rec_B, self.data_B)

        # Backpropagation
        self.loss_G = self.loss_G_AtoB + self.loss_G_BtoA \
                 + self.loss_cycle_AtoB * self.lambdaA + self.loss_cycle_BtoA * self.lambdaB

        if self.lambda_idt:
            self.idt_A = self.netG_AtoB(self.data_B)
            self.idt_B = self.netG_BtoA(self.data_A)
            self.loss_idt_A = self.cri_idt(self.idt_A, self.data_B) * self.lambda_idt * self.lambdaB
            self.loss_idt_B = self.cri_idt(self.idt_B, self.data_A) * self.lambda_idt * self.lambdaA
            self.loss_G += self.loss_idt_A + self.loss_idt_B

        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()

        # -------------------------- train discriminator D_A --------------------------

        self.set_requires_grad([self.netD_A, self.netD_B], requires_grad=True)

        disc_real_A = self.netD_A(self.data_A)
        loss_D_A_real = self.cri_gan(disc_real_A, torch.ones_like(disc_real_A).to(self.device))

        fake_A = self.fake_A_pool.query(self.fake_A)
        disc_fake_A = self.netD_A(fake_A.detach())
        loss_D_A_fake = self.cri_gan(disc_fake_A, torch.zeros_like(disc_fake_A).to(self.device))

        self.loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

        # -------------------------- train discriminator D_B --------------------------

        disc_real_B = self.netD_B(self.data_B)
        loss_D_B_real = self.cri_gan(disc_real_B, torch.ones_like(disc_real_B).to(self.device))

        fake_B = self.fake_B_pool.query(self.fake_B)
        disc_fake_B = self.netD_B(fake_B.detach())
        loss_D_B_fake = self.cri_gan(disc_fake_B, torch.zeros_like(disc_fake_B).to(self.device))

        self.loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
        self.loss_D = self.loss_D_A + self.loss_D_B

        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

    def lr_scheduler_lambda(self, epoch):
        return 1.0 - max(0, epoch + - self.opt['Train']['lr_init_n_epochs'] + 1) \
               / float(self.opt['Train']['lr_decay_n_epochs'] + 1)

    def test(self):
        with torch.no_grad():
            self.forward()
