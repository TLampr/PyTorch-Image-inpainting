import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import time
from PIL import Image
import matplotlib.pyplot as plt
#pic = plt.imread('rick_morty_erase.png')
from torchvision import models

im_frame = Image.open('rick_morty_jpg_erase.jpg', 'r')
has_alpha = im_frame.mode == 'RGBA'
red, green, blue, alpha = im_frame.split()
plt.imshow(alpha)
plt.show()
print("hola")

vgg16 = models.vgg16(pretrained=True)

# random_image = []
# for x in range(0 , 512*512):
#         random_image.append(np.random.randint(0 , 255))
#
#
# random_image_arr = np.array(random_image)
# image_d = torch.FloatTensor(np.asarray(random_image_arr.reshape(1, 1, 512 , 512)))
# conv2 = torch.nn.Conv2d(1, 18, kernel_size=7, stride=2, padding=0)
# cosa = conv2(image_d)
# print("hola")
