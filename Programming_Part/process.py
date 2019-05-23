from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import torchvision
import torch
from tqdm import tqdm

dir_path = "./destroyed"


destroyed_images = []
destroyed_images2 = []
for infile in tqdm(glob.glob(dir_path + "/*.png")):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    to_Pill = torchvision.transforms.ToPILImage()
    to_Tensor = torchvision.transforms.ToTensor()
    im = np.transpose(np.array(im),(2,0,1))
    tensor_im = to_Tensor(im)
    tensor_im = tensor_im.transpose(0,1).transpose(1,2)
    destroyed_images.append(tensor_im)

destroyed_images = torch.stack(destroyed_images)
torch.save(destroyed_images,'total_dest_data.pt')

dir_path = "./complete"
destroyed_images = []
for infile in tqdm(glob.glob(dir_path + "/*.png")):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    to_Pill = torchvision.transforms.ToPILImage()
    to_Tensor = torchvision.transforms.ToTensor()
    im = np.transpose(np.array(im),(2,0,1))
    tensor_im = to_Tensor(im)
    tensor_im = tensor_im.transpose(1,0).transpose(1,2)
    destroyed_images.append(tensor_im)

destroyed_images = torch.stack(destroyed_images)
torch.save(destroyed_images,'data.pt')
