import torch
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import transforms
import torchvision.models as models


class loss(nn.modules.loss._WeightedLoss):

    def __init__(self):
        super(loss, self).__init__()
        self.C = 3
        self.H = 512
        self.W = 512
        self.N = self.C*self.H*self.W
        self.outputs = []
        self.Igt = None
        self.Iout = None
        self.mask = None
        self.Icomp = None
        
    def forward(self, Igt, Iout, mask) :
        self.Igt = Igt
        self.Iout = Iout
        self.mask = mask
        self.make_comp()
        return self.loss_function()
    
    def make_comp(self) :
        self.Icomp = self.Igt.clone()
        mask = self.mask.clone().type(torch.ByteTensor)
        mask[mask!=0] = 1        
        for i in range(self.Igt.shape[0]):
            for j in range(self.Igt.shape[1]):
                self.Icomp[i, j, :, :][mask[i, 0,]] = self.Igt[i ,j, :, :][mask[i, 0] == 1]       
        """mask = np.ones(X.shape, dtype=bool)
        mask[:, validation_ind] = 0

        X_train = X[mask].reshape(d*n_len, -1)"""

    def loss_function(self):
        loss = 6*self.l_hole() + self.l_valid() + 0.05*self.l_perc() \
               + 120*(self.l_style_comp() + self.l_style_out()) + 0.1*self.l_tv()
        return loss

    def l_hole(self):
        """
        Computation of one of the pixel losses: l_hole
        :return: value of l_hole
        """        
        Nigt = self.N
        aux1 = (1 - self.mask)
        aux2 = self.Iout - self.Igt
        aux3 = torch.zeros((3, 3, 512, 512))
        for i in range(self.mask.shape[0]):
            aux3 += aux1[i, 0] * aux2[i]
        l1_loss = torch.norm(aux3, p=1)
        l_hole = l1_loss/Nigt
        return l_hole

    def l_valid(self):
        """
        Computation of one of the pixel losses: l_valid
        :return: value of l_valid
        """
        Nigt = self.N
        aux1 = self.Iout - self.Igt
        aux2 = torch.zeros((3, 3, 512, 512))
        for i in range(self.mask.shape[0]):
            aux2 += self.mask[i, 0] * aux1[i]
        l1_loss = torch.norm(aux2, p=1)
        l_valid = l1_loss/Nigt
        return l_valid

    def l_perc(self):
        """
        Computation of perceptual loss
        :return: value of l_perc
        """
        vgg16 = models.vgg16(pretrained=True)
        #min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        
        """
        transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        """
        
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)
        
        # out
        self.outputs = []
        #img = transform_pipeline(self.Iout)
        #img = img.unsqueeze(0)
        img = Variable(self.Iout)
        out = vgg16.forward(img)
        self.pool_out = self.outputs.copy()

        # gt
        self.outputs = []
        #img = transform_pipeline(self.Igt)
        #img = img.unsqueeze(0)
        img = Variable(self.Igt)
        out = vgg16.forward(img)
        self.pool_gt = self.outputs.copy()

        # comp
        self.outputs = []
        #img = transform_pipeline(self.Icomp)
        #img = img.unsqueeze(0)
        img = Variable(self.Icomp)
        out = vgg16.forward(img)
        self.pool_comp = self.outputs.copy()

        l_perc = 0.0
        for i in range(len(self.pool_out)):
            Nigt = torch.tensor(self.pool_gt[0].shape).prod()
            diff1 = self.pool_out[i] - self.pool_gt[i]
            l1_loss1 = torch.norm(diff1, p=1)
            diff2 = self.pool_comp[i] - self.pool_gt[i]
            l1_loss2 = torch.norm(diff2, p=1)
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
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1/(C * H * W)
            for b in range(B):
                for c in range(C):
                    phi_out_t = torch.transpose(self.pool_out[i][b,c,:,:], 0, 1)
                    phi_gt_t = torch.transpose(self.pool_gt[i][b,c,:,:], 0, 1)
                    diff = Kp * (torch.mm(phi_out_t, self.pool_out[i][b,c,:,:]) - torch.mm(phi_gt_t, self.pool_gt[i][b,c,:,:]))
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
            Kp = 1/(C * H * W)
            for b in range(B):
                for c in range(C):
                    phi_comp_t = torch.transpose(self.pool_comp[i][b,c,:,:], 0, 1)
                    phi_gt_t = torch.transpose(self.pool_gt[i][b,c,:,:], 0, 1)
                    diff = Kp * (torch.mm(phi_comp_t, self.pool_comp[i][b,c,:,:]) -
                                 torch.mm(phi_gt_t, self.pool_gt[i][b,c,:,:]))
                    l1 = torch.norm(diff, p=1)
                    l_style += l1 / (C * C)
        return l_style
    
    def l_tv(self):
        """
        Compute the smoothing penalty: total variation loss
        :return: total variation value
        """
        B, C, H, W = self.Icomp.shape
        Ncomp = C * H * W
        aux1 = 0.0
        aux2 = 0.0
        for b in range(B):
            for i in range(H):
                for j in range(W-1):
                    if j < W - 1:
                        diff = self.Icomp[b,:,i,j+1] - self.Icomp[b,:,i,j]
                        aux1 += torch.norm(diff, p=1)
                    if i < H - 1:
                        diff = self.Icomp[b, :, i + 1, j] - self.Icomp[b, :, i, j]
                        aux2 += torch.norm(diff, p=1)
        l_tv = (aux1 + aux2)/Ncomp
        return l_tv

# if __name__ == '__main__':
#     lt = loss_test(Igt, Iout, mask)
#     lt.img = Image.open('rick_morty.png', 'r')
#     lt.l_perc()
