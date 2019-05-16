import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Util(object):
    def __init__(self, args):
        self.args = args

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def random_bbox(self):
        img_shape = self.args[0]
        img_height = img_shape[0]
        img_width = img_shape[1]

        maxt = img_height - self.args[3] - self.args[1]
        maxl = img_width - self.args[4] - self.args[2]

        t = randint(self.args[3], maxt)
        l = randint(self.args[4], maxl)
        h = self.args[1]
        w = self.args[2]
        return (t, l, h, w)

    def bbox2mask(self, bbox):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [B, 1, H, W]
        """

        def npmask(bbox, height, width, delta_h, delta_w):
            mask = np.zeros((1, 1, height, width), np.float32)
            h = np.random.randint(delta_h // 2 + 1)
            w = np.random.randint(delta_w // 2 + 1)
            mask[:, :, bbox[0] + h: bbox[0] + bbox[2] - h,
            bbox[1] + w: bbox[1] + bbox[3] - w] = 1.
            return mask

        img_shape = self.args[0]
        height = img_shape[0]
        width = img_shape[1]

        mask = npmask(bbox, height, width,
                      self.args[5],
                      self.args[6])

        return torch.FloatTensor(mask)

    def local_patch(self, x, bbox):
        '''
        bbox[0]: top
        bbox[1]: left
        bbox[2]: height
        bbox[3]: width
        '''
        x = x[:, :, bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]]

        return x


if __name__ is '__main__':
    image_shape = (512, 512)
    mask_height = 400
    mask_width = 400
    mask_vertical_margin = 10
    mask_horizontal_margin = 10
    max_delta_height = image_shape[0] // 8
    max_delta_width = image_shape[0] // 8
    args = (image_shape, mask_height, mask_width, mask_vertical_margin, mask_horizontal_margin, max_delta_height,
            max_delta_width)
    utils = Util(args=args)
    bbox = utils.random_bbox()
    mask = utils.bbox2mask(bbox=bbox)
    print('hi')
