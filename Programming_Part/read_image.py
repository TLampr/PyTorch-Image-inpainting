from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import torchvision
import torch
import tqdm

dir_path = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png"
dir_path2 = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/masks"
masks = {}
"""THE MASKS"""
for infile in glob.glob(dir_path2 + "/*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    trans = torchvision.transforms.ToPILImage()
    trans2 = torchvision.transforms.ToTensor()
    alpha_im = np.array(trans(trans2(im)).split()[3])
    alpha_im[alpha_im != 255] = 0
    name = infile.split('/')[-1]
    masks[name] = trans(trans2(alpha_im))

torch.save(masks, 'masks.pt')

key_array = np.array(list(masks.keys()))
dest_tensor_list = []
tensor_list = []
total_dest_tensor_list = []
counter = 0
"""SAVES THE IMAGES AND THE DESTROYED ONES AS TWO BIG TENSORS"""
for infile in glob.glob(dir_path + "/*.png"):
    file, ext = os.path.splitext(infile)
    """READS THE IMAGE"""
    im = Image.open(infile)
    """PIL TRANSFORMATION"""
    trans = torchvision.transforms.ToPILImage()
    """TENSOR TRANSFORMATION"""
    trans2 = torchvision.transforms.ToTensor()
    tensor_im = trans2(im)
    pil_im = trans(trans2(im))
    pil_im_dest = trans(trans2(im))
    """GET A RANDOM DESTRUCTION PATTERN"""
    key = np.random.choice(key_array)
    mask = np.array(masks[key])
    """MAKE IT 1/0"""
    mask[mask != 0] = 1
    """CREATE AN IMAGE WITH ALL RGB CHANNELS AFFECTED"""
    pil_im_total_dest = trans(np.multiply(np.array(pil_im_dest), mask[:, :, None]))
    pil_im_total_dest.putalpha(masks[key])
    """CREATE AN IMAGE WITH ONLY THE ALPHA CHANNEL DESTROYED"""
    pil_im_dest.putalpha(masks[key])
    tensor_im_total_dest = trans2(pil_im_total_dest)
    tensor_im_dest = trans2(pil_im_dest)
    dest_tensor_list.append(tensor_im_dest)
    tensor_list.append(tensor_im)
    total_dest_tensor_list.append(tensor_im_total_dest)
    counter += 1
    """STOP AT 10 IMAGES"""
    if counter == 10:
        break
"""STACK THE DATA IN (NUMBER x CHANNELS x PIXELS x PIXELS)"""
dest_data = torch.stack(dest_tensor_list)
data = torch.stack(tensor_list)
total_dest_data = torch.stack(total_dest_tensor_list)
"""SAVE THE DATA IN PT FORMAT"""
torch.save(dest_data, 'dest_data.pt')
torch.save(data, 'data.pt')
torch.save(total_dest_data, 'total_dest_data.pt')
