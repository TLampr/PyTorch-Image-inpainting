# squared images
from PIL import Image
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from loss import loss


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

        i = int(np.log2(512 / img_shape))

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
                nn.Conv2d(3, self.nb_channels[0], kernel_size=self.kernel_sizes[0], stride=2,
                          padding=int((self.kernel_sizes[0] - 1) / 2)), nn.ReLU()))

        for j in range(1, self.nb_layers):
            self.encoding_list.append(
                nn.Sequential(
                    nn.Conv2d(self.nb_channels[j - 1], self.nb_channels[j],
                              kernel_size=self.kernel_sizes[j], stride=2,
                              padding=int((self.kernel_sizes[j] - 1) / 2)),
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
                    nn.Conv2d(self.nb_channels[self.nb_layers - j - 1] + self.nb_channels[self.nb_layers - j - 2],
                              self.nb_channels[self.nb_layers - j - 2], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.nb_channels[self.nb_layers - j - 2]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.decoding_list.append(
            nn.Sequential(
                nn.Conv2d(self.nb_channels[0] + 3, 3, kernel_size=3, stride=1, padding=1))
        )
        # no batch norm or relu here -> output is the reconstructed image
        # print(self.encoding_list)
        # print(self.decoding_list)

    def forward(self, x):
        output_feature = []
        out = x
        output_feature.append(out)
        for j in range(self.nb_layers):
            out = self.encoding_list[j](out)
            output_feature.append(out)

        # torch.cat((first_tensor, second_tensor), dimension)

        for j in range(self.nb_layers):
            nearestUpSample = nn.UpsamplingNearest2d(scale_factor=2)(out)
            concat = torch.cat((output_feature[self.nb_layers - j - 1], nearestUpSample), dim=1)
            out = self.decoding_list[j](concat)

        return out


if __name__ == '__main__':

    img512 = torch.rand(10, 4, 512, 512)
    img256 = torch.rand(10, 4, 256, 256)
    img128 = torch.rand(10, 4, 128, 128)
    img64 = torch.rand(10, 4, 64, 64)
    img32 = torch.rand(10, 4, 32, 32)

    list_img = []
    list_img.append(img512)
    list_img.append(img256)
    list_img.append(img128)
    list_img.append(img64)
    list_img.append(img32)

    for img in list_img:
        img_shape = img.shape[2]
        nb_jump = 3
        i = int(np.log2(512 / img_shape))

        print("starting layer : ", i)
        print("nb of jumps : ", nb_jump)

        nb_layers = 8 - i
        nb_channels = []

        size = img_shape
        for j in range(nb_jump):
            size = int(size / 2)
            nb_channels.insert(0, size)

        for j in range(nb_layers - nb_jump):
            nb_channels.append(img_shape)

        print("nb de layers encoding et decoding : ", nb_layers)
        print("check size : ", len(nb_channels))
        print("nb of channels in the network : ", nb_channels)

        model = UNet(img_shape)
        output = model(img)

        #mask = make_mask()
        loss = loss(img, output)

        output_feature = []
        out = img
        output_feature.append(out)
        for j in range(nb_layers):
            out = model.encoding_list[j](out)
            output_feature.append(out)
            print(out.shape)

        for j in range(nb_layers):
            nearestUpSample = nn.UpsamplingNearest2d(scale_factor=2)(out)
            concat = torch.cat((output_feature[nb_layers - j - 1], nearestUpSample), dim=1)
            out = model.decoding_list[j](concat)
            print(out.shape)

        # print(model)
