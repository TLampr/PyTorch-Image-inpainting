import torch
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import transforms
import torchvision.models as models


class loss:

    def __init__(self, Igt, Iout, mask):
        self.C = 3
        self.H = 512
        self.W = 512
        self.N = self.C*self.H*self.W
        self.outputs = []
        self.Igt = Igt
        self.Iout = Iout
        self.mask = mask

    def loss_function(self):
        loss = self.l_hole() + self.l_valid() + self.l_perc() + self.l_style_comp() + self.l_style_out()
        return loss

    def l_hole(self):
        """
        Computation of one of the pixel losses: l_hole
        :return: value of l_hole
        """
        Nigt = self.N
        aux1 = (1 - self.mask)
        aux2 = self.Iout - self.Igt
        aux3 = (aux1*aux2).view(-1)
        l1_loss = torch.norm(aux3, p=1, dim=1)
        l_hole = l1_loss/Nigt
        return l_hole

    def l_valid(self):
        """
        Computation of one of the pixel losses: l_valid
        :return: value of l_valid
        """
        Nigt = self.N
        aux1 = self.Iout - self.Igt
        aux2 = self.mask*aux1
        l1_loss = torch.norm(aux2.view(-1), p=1, dim=1)
        l_valid = l1_loss/Nigt
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
        # out
        img = transform_pipeline(self.Iout)
        img = img.unsqueeze(0)
        img = Variable(img)
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)
        out = vgg16.forward(img)
        self.pool_out = self.outputs.clone()

        # gt
        self.outputs = []
        img = transform_pipeline(self.Igt)
        img = img.unsqueeze(0)
        img = Variable(img)
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)
        out = vgg16.forward(img)
        self.pool_gt = self.outputs.clone()

        # comp
        self.outputs = []
        img = transform_pipeline(self.Icomp)
        img = img.unsqueeze(0)
        img = Variable(img)
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)
        out = vgg16.forward(img)
        self.pool_comp = self.outputs.clone()

        l_perc = 0.0
        for i in range(len(self.pool_out)):
            Nigt = torch.tensor(self.pool_gt[0].shape).prod()
            diff1 = self.pool_out[i] - self.pool_gt[i]
            l1_loss1 = torch.norm(diff1.view(-1), p=1, dim=1)
            diff2 = self.pool_comp[i] - self.pool_gt[i]
            l1_loss2 = torch.norm(diff2.view(-1), p=1, dim=1)
            l_perc += (l1_loss1 + l1_loss2)/Nigt
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
            C, H, W = self.pool_gt.shape
            Kp = C * H * W
            phi_out_t = torch.transpose(self.pool_out, 0, 1)
            phi_gt_t = torch.transpose(self.pool_gt, 0, 1)
            diff = Kp*(torch.mm(phi_out_t, self.pool_out) - torch.mm(phi_gt_t, self.pool_gt))
            l1 = torch.norm(diff.view(-1), p=1, dim=1)
            l_style += l1/(C*C)
        return l_style

    def l_style_comp(self):
        """
        Compute style loss term: style comp loss
        :return: value style comp loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            C, H, W = self.pool_gt.shape
            Kp = C * H * W
            phi_comp_t = torch.transpose(self.pool_comp, 0, 1)
            phi_gt_t = torch.transpose(self.pool_gt, 0, 1)
            diff = Kp*(torch.mm(phi_comp_t, self.pool_comp) - torch.mm(phi_gt_t, self.pool_gt))
            l1 = torch.norm(diff.view(-1), p=1, dim=1)
            l_style += l1/(C*C)
        return l_style

# if __name__ == '__main__':
#     lt = loss_test(Igt, Iout, mask)
#     lt.img = Image.open('rick_morty.png', 'r')
#     lt.l_perc()
