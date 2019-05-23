import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import glob, os
from tqdm import tqdm
from torchvision import models
import torchvision.transforms as transforms
# import objgraph
import gc
import matplotlib.pylab as plt


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()

        # this is the main convolution, which is going to be in charge of
        # getting the 2D convolution of the element-wise multiplication
        # between mask and input
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        torch.nn.init.xavier_uniform_(self.input_conv.weight)

        # it is needed to compute the convolution of the mask separately
        # in order to update its size in each level of the u-net

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # conv is updated
        for param in self.input_conv.parameters():
            param.requires_grad = True

    def forward(self, args):

        image, mask = args

        output = self.input_conv(image * mask)

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        zeros_mask = output_mask == 0
        sum_M = output_mask.masked_fill_(zeros_mask, 1.0)

        output_pre = (output) / sum_M + output_bias
        output = output_pre.masked_fill_(zeros_mask, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(zeros_mask, 0.0)

        del image, mask, output_pre, sum_M, zeros_mask, output_bias

        return output, new_mask


class UNet(nn.Module):
    def __init__(self, img_shape):
        super(UNet, self).__init__()

        # Todo : nb_jump = get_params()
        self.get_params(img_shape)

        # ENCODING :
        self.encoding()

        # DECODING :
        self.decoding()

    def get_params(self, img_shape):
        """
        This function is in charge of setting all the params
        :return:
        """
        if img_shape == 512:
            self.nb_jump = 3
        elif img_shape == 256:
            self.nb_jump = 3
        elif img_shape == 128:
            self.nb_jump = 3
        elif img_shape == 64:
            self.nb_jump = 3
        elif img_shape == 32:
            self.nb_jump = 3
        else:
            print("error in image size")

        i = int(np.log2(256 / img_shape))

        self.kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]

        self.nb_layers = 8 - i
        self.nb_channels = []

        size = img_shape
        for j in range(self.nb_jump):
            size = int(size / 2)
            self.nb_channels.insert(0, size)

        for j in range(self.nb_layers - self.nb_jump):
            self.nb_channels.append(img_shape)

    def encoding(self):
        """
        Function in charge of the encoding part of the U-Net
        :return:
        """
        self.encoding_list = nn.ModuleList()
        self.encoding_list.append(
            nn.Sequential(
                PartialConv2d(3, self.nb_channels[0], kernel_size=self.kernel_sizes[0], stride=2,
                              padding=int((self.kernel_sizes[0] - 1) / 2), bias=False), nn.ReLU()))

        for j in range(1, self.nb_layers):
            self.encoding_list.append(
                nn.Sequential(
                    PartialConv2d(self.nb_channels[j - 1], self.nb_channels[j],
                                  kernel_size=self.kernel_sizes[j], stride=2,
                                  padding=int((self.kernel_sizes[j] - 1) / 2), bias=False),
                    nn.BatchNorm2d(self.nb_channels[j]),
                    nn.ReLU()))

    def decoding(self):
        """
        Function in charge of executing the decoding part of the U-Net
        :return:
        """
        self.decoding_list = nn.ModuleList()
        for j in range(self.nb_layers - 1):
            self.decoding_list.append(
                nn.Sequential(
                    PartialConv2d(self.nb_channels[self.nb_layers - j - 1] + self.nb_channels[self.nb_layers - j - 2],
                                  self.nb_channels[self.nb_layers - j - 2], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.nb_channels[self.nb_layers - j - 2]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.decoding_list.append(
            nn.Sequential(
                PartialConv2d(self.nb_channels[0] + 3, 3, kernel_size=3, stride=1, padding=1))
        )

    def forward(self, x, mask):
        output_feature = []
        out = x, mask
        output_feature.append(out)
        for j in range(self.nb_layers):
            out = self.encoding_list[j][0](out)
            image = self.encoding_list[j][1](out[0])
            out = image, out[1]
            output_feature.append(out)

        # torch.cat((first_tensor, second_tensor), dimension)

        for j in range(self.nb_layers):
            nearestUpSample_image = nn.UpsamplingNearest2d(scale_factor=2)(out[0])
            nearestUpSample_mask = nn.UpsamplingNearest2d(scale_factor=2)(out[1])

            concat_image = torch.cat((output_feature[self.nb_layers - j - 1][0], nearestUpSample_image), dim=1)
            concat_mask = torch.cat((output_feature[self.nb_layers - j - 1][1], nearestUpSample_mask), dim=1)

            out = concat_image, concat_mask
            out = self.decoding_list[j][0](out)

            if j < self.nb_layers - 1:
                image = self.decoding_list[j][1](out[0])
                out = image, out[1]
        del output_feature, image, nearestUpSample_image, nearestUpSample_mask, concat_image, concat_mask
        return out

if __name__=="__main__":
    model = UNet(256)
    # model.load_state_dict(torch.load('./noreg_checkpoint_last_loss_epoch44.pt', map_location='cpu'))
    model.load_state_dict(torch.load('./short_name_checkpoint9.pt', map_location='cpu'))

    model.eval()

    import torchvision

    im = torch.load('./total_dest_data.pt')

    data = im[:,:3,:,:]
    mask = im[:,3,:,:][:,None,:,:]
    masks = torch.cat((mask,mask,mask),dim=1)









    rec = model(data, masks)
    output_image = np.transpose(np.array(rec[0].detach()),(0,2,3,1))
    """0,1,2,3,4"""
    """Shape: d,d,c"""
    plt.imshow(output_image[1])
    plt.show()
    print('hi')



















