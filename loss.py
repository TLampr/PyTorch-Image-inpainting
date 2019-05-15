import torch
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np


class loss(nn.Module):

    def __init__(self, img, output, mask):
        super(loss, self).__init__()
        self.C = 3
        self.H = 512
        self.W = 512
        self.outputs = []
        self.Igt = img  # Igt
        self.Iout = output.data  # Iout torch.nn.Parameter()
        self.mask = mask  # [None, :, :, :]
        self.Icomp = self.make_comp()
        self.N = self.C * self.H * self.W * self.Igt.shape[0]

    def make_comp(self):
        self.Icomp = self.Iout.clone()
        self.mask[self.mask != 0] = 1
        dummy = self.mask.clone().long()
        for b in range(self.Igt.shape[0]):
            for c in range(self.Igt.shape[1]):
                self.Icomp[b, c, :, :][np.where(dummy[b, 0, :, :] == 1)] = self.Igt[b, c, :, :][
                    np.where(dummy[b, 0, :, :] == 1)]
        return self.Icomp

    def loss_function(self):
        loss = Variable(self.l_hole(), requires_grad=True) + Variable(self.l_valid(), requires_grad=True) + Variable(
            self.l_perc(), requires_grad=True) + Variable(self.l_style_comp(), requires_grad=True) + Variable(
            self.l_style_out(), requires_grad=True)
        return loss


    def l_hole(self):
        """
        Computation of one of the pixel losses: l_hole
        :return: value of l_hole
        """
        Nigt = self.N
        aux1 = (1 - self.mask)
        aux2 = self.Iout - self.Igt
        l1_loss = 0.0
        for i in range(aux2.shape[0]):
            l1_loss += torch.norm(aux1 * aux2, p=1)
        l_hole = l1_loss / Nigt
        return l_hole

    def l_valid(self):
        """
        Computation of one of the pixel losses: l_valid
        :return: value of l_valid
        """
        Nigt = self.N
        aux1 = self.Iout - self.Igt
        l1_loss = 0.0
        for i in range(self.mask.shape[0]):
            aux2 = self.mask[i][None, :, :, :] * aux1[i][None, :, :, :]
            l1_loss += torch.norm(aux2, p=1)
        l_valid = l1_loss / Nigt
        return l_valid

    def l_perc(self):
        """
        Computation of perceptual loss
        :return: value of l_perc
        """
        vgg16 = models.vgg16(pretrained=True)
        min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)

        # out
        img = Variable(self.Iout)
        out = vgg16.forward(img)

        # gt
        img = Variable(self.Igt)
        out = vgg16.forward(img)

        # comp
        img = Variable(self.Icomp)
        out = vgg16.forward(img)

        self.pool_out = self.outputs[0:3].copy()
        self.pool_gt = self.outputs[3:6].copy()
        self.pool_comp = self.outputs[6:9].copy()
        l_perc = 0.0
        for i in range(len(self.pool_out)):
            Nigt = torch.tensor(self.pool_gt[0].shape).prod()
            diff1 = self.pool_out[i] - self.pool_gt[i]
            l1_loss1 = torch.norm(diff1, p=1)
            diff2 = self.pool_comp[i] - self.pool_gt[i]
            l1_loss2 = torch.norm(diff2, p=1)
            l_perc += (l1_loss1 + l1_loss2) / Nigt
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
            Kp = 1 / (C * H * W)
            for b in range(B):
                for c in range(C):
                    phi_out_t = torch.transpose(self.pool_out[i][b, c, :, :], 0, 1)
                    phi_gt_t = torch.transpose(self.pool_gt[i][b, c, :, :], 0, 1)
                    diff = Kp * (torch.mm(phi_out_t, self.pool_out[i][b, c, :, :]) - torch.mm(phi_gt_t,
                                                                                              self.pool_gt[i][b, c, :,
                                                                                              :]))
                    l1 = torch.norm(diff, p=1)
                    l_style += l1 / (C * C)
        return l_style

    def l_style_comp(self):
        """
        Compute style loss term: style comp loss
        :return: value style comp loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1 / (C * H * W)
            for b in range(B):
                for c in range(C):
                    phi_comp_t = torch.transpose(self.pool_comp[i][b, c, :, :], 0, 1)
                    phi_gt_t = torch.transpose(self.pool_gt[i][b, c, :, :], 0, 1)
                    diff = Kp * (torch.mm(phi_comp_t, self.pool_comp[i][b, c, :, :]) -
                                 torch.mm(phi_gt_t, self.pool_gt[i][b, c, :, :]))
                    l1 = torch.norm(diff, p=1)
                    l_style += l1 / (C * C)
        return l_style

# if __name__ == '__main__':
#     lt = loss_test(Igt, Iout, mask)
#     lt.img = Image.open('rick_morty.png', 'r')
#     lt.l_perc()
