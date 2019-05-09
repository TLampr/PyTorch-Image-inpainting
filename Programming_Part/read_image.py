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

key_array = np.array(list(masks.keys()))
dest_tensor_list = []
tensor_list = []
counter = 0
"""SAVES THE IMAGES AND THE DESTROYED ONES AS TWO BIG TENSORS"""
for infile in glob.glob(dir_path + "/*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    trans = torchvision.transforms.ToPILImage()
    trans2 = torchvision.transforms.ToTensor()
    plt.imshow(trans(trans2(im)))
    tensor_im = trans2(im)
    pil_im = trans(trans2(im))
    pil_im_dest = trans(trans2(im))
    key = np.random.choice(key_array)
    pil_im_dest.putalpha(masks[key])
    tensor_im_dest = trans2(pil_im_dest)
    plt.imshow(trans(trans2(pil_im_dest)))
    dest_tensor_list.append(tensor_im_dest)
    tensor_list.append(tensor_im)
    counter += 1
    """STOP AT 10 IMAGES"""
    if counter == 10:
        break
dest_data = torch.stack(dest_tensor_list)
data = torch.stack(tensor_list)
torch.save(dest_data, 'dest_data.pt')
torch.save(data, 'data.pt')
