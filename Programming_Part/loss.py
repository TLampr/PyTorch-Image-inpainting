import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models

is_cuda = torch.cuda.is_available()

class loss(nn.modules.loss._WeightedLoss):
    def __init__(self):
        super(loss, self).__init__()
        self.C = 3
        self.H = 256
        self.W = 256
        self.N = self.C * self.H * self.W
        self.outputs = []
        self.Igt = None
        self.Iout = None
        self.mask = None
        self.Icomp = None

    def forward(self, Igt, Iout, mask):
        self.Igt = Igt
        self.Iout = Iout
        self.mask = mask
        self.make_comp()
        
        return self.loss_function()

    def make_comp(self):
        mask = self.mask.clone().type(torch.ByteTensor)
        if is_cuda :
            mask = mask.cuda() 
            
        self.Icomp = Variable(torch.where(mask, self.Igt, self.Iout), requires_grad = True)

    def loss_function(self):
        loss_total = 6 * self.l_hole() + self.l_valid() + 0.05 * self.l_perc()\
                    + 120 * (self.l_style_comp() + self.l_style_out()) + 0.1 * self.l_tv()

        self.Igt = None
        self.Iout = None
        self.Icomp = None
        self.mask = None
        self.pool_out = None
        self.pool_gt = None
        self.pool_comp = None
        
        if is_cuda :
            torch.cuda.empty_cache()

        return loss_total

    def l_hole(self):
        """
        Computation of one of the pixel losses: l_hole
        :return: value of l_hole
        """

        B, C, W, H = self.Igt.shape
        Nigt = self.N*B
        aux1 = (1 - self.mask)
        aux2 = self.Iout - self.Igt
        l_hole = torch.norm((aux1*aux2), p=1)/Nigt
        
        del aux1
        del aux2
        
        if is_cuda :
            torch.cuda.empty_cache()
            
        return l_hole

    def l_valid(self):
        """
        Computation of one of the pixel losses: l_valid
        :return: value of l_valid
        """
        B, C, W, H = self.Igt.shape

        Nigt = self.N * B
        aux1 = self.Iout - self.Igt
        l_valid = torch.norm((aux1*self.mask), p=1)/Nigt
        
        del aux1
        
        if is_cuda : 
            torch.cuda.empty_cache()
            
        return l_valid

    def l_perc(self):
        """
        Computation of perceptual loss
        min_img_size = 224
        The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        :return: value of l_perc
        """

        vgg16 = models.vgg16(pretrained=True).cuda()
        for param in vgg16.features.parameters():
            param.requires_grad = False
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)

        self.outputs = []

        img = Variable(self.Iout, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_out = list(self.outputs)

        self.outputs = []

        img = Variable(self.Igt, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_gt = list(self.outputs)

        # comp
        self.outputs = []

        img = Variable(self.Icomp, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_comp = list(self.outputs)

        l_perc = 0.0
        for i in range(len(self.pool_out)):
            Nigt = torch.tensor(self.pool_gt[i].shape).prod()
            diff1 = self.pool_out[i] - self.pool_gt[i]
            l1_loss1 = torch.norm(diff1, p=1)
            diff2 = self.pool_comp[i] - self.pool_gt[i]
            l1_loss2 = torch.norm(diff2, p=1)
            l_perc += (l1_loss1 + l1_loss2)/Nigt

        del img
        del out
        del Nigt
        del diff1
        del diff2
        del l1_loss2
        del l1_loss1
        self.outputs = []
        
        if is_cuda :
            torch.cuda.empty_cache()

        return l_perc

    def hook(self, module, input, output):
        self.outputs.append(output)

    def l_style_out(self):
        """
        Compute style loss term: style out loss
        :return: value style out loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1.0 / (C * H * W* B)
            aux1 = torch.matmul(torch.transpose(self.pool_out[i],2,3),self.pool_out[i])
            aux2 = torch.matmul(torch.transpose(self.pool_gt[i],2,3),self.pool_gt[i])
            l_1 = torch.norm((aux1-aux2)*Kp, p=1)/(C**2)
            l_style += l_1

        del aux1
        del aux2
        del l_1
        
        if is_cuda :
            torch.cuda.empty_cache()

        return l_style

    def l_style_comp(self):
        """
        Compute style loss term: style comp loss
        :return: value style comp loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1.0 / (C * H * W*B)
            aux1 = torch.matmul(torch.transpose(self.pool_comp[i], 2, 3), self.pool_comp[i])
            aux2 = torch.matmul(torch.transpose(self.pool_gt[i], 2, 3), self.pool_gt[i])
            l_1 = torch.norm((aux1 - aux2) * Kp, p=1) / (C ** 2)
            l_style += l_1
            
        del aux1
        del aux2
        del l_1

        if is_cuda :
            torch.cuda.empty_cache()

        return l_style

    def l_tv(self):
        """
        Compute the smoothing penalty: total variation loss
        :return: total variation value
        """
        B, C, H, W = self.Icomp.shape
        Ncomp = B * C * H * W
        
        aux1 = self.Icomp[:, :, :, :-1] - self.Icomp[:, :, :, 1:]
        aux2 = self.Icomp[:, :, :-1, :] - self.Icomp[:, :, 1:, :]
        
        l1_loss = torch.norm(aux1[self.mask[:, :, :, :-1] == 1], p=1) + torch.norm(aux2[self.mask[:, :, :-1, :] == 1], p=1)
        l_tv = l1_loss/Ncomp

        del aux1, aux2, l1_loss
        
        if is_cuda :
            torch.cuda.empty_cache()

        return l_tv
