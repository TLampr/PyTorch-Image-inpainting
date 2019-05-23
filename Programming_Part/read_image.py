from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import torchvision
import torch
from tqdm import tqdm

dir_path = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed"
#"C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png"
dir_path2 = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed/masks"
#"C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/masks"
dir_path3 = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed"
#"C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed"
masks = {}
masks_tensors = {}
"""THE MASKS"""

masks = {}
masks_tensors = {}
original_masks = {}
original_tensors = {}

"""THE MASKS"""
for infile in tqdm(glob.glob(dir_path2 + "/*.png")):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im = im.resize((256,256),Image.ANTIALIAS)
    trans = torchvision.transforms.ToPILImage()
    trans2 = torchvision.transforms.ToTensor()
    alpha_im = np.array(trans(trans2(im)).split()[3])
    original_alpha = alpha_im.copy()
    alpha_im[alpha_im != 255] = 0
    alpha_im[alpha_im == 255] = 1
    name = infile.split('/')[-1]
    masks[name] = trans(trans2(alpha_im))
    original_masks[name] = trans(trans2(original_alpha))
    original_tensors[name] = trans2(original_alpha)
    masks_tensors[name] = trans2(alpha_im)
torch.save(masks_tensors, 'masks.pt')
torch.save(original_tensors, 'original_masks.pt')

key_array = np.array(list(masks.keys()))
dest_tensor_list = []
tensor_list = []
total_dest_tensor_list = []
counter = 0
data_counter = 0
# masks = torch.load("masks.pt")
# original_masks = torch.load('original_masks.pt')
"""SAVES THE IMAGES AND THE DESTROYED ONES AS TWO BIG TENSORS"""
for infile in tqdm(glob.glob(dir_path + "/*.png")):
    file, ext = os.path.splitext(infile)
    """READS THE IMAGE"""
    im = Image.open(infile)
    im = im.resize((256,256),Image.ANTIALIAS)
    im.save('image{}.png'.format(counter))
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
    original_pitd = pil_im_total_dest.copy()
    secondary_pitd = pil_im_total_dest.copy()
    pil_im_total_dest.putalpha(masks[key])
    original_pitd.putalpha(original_masks[key])
    original_pitd.save('destroyed_image{}.png'.format(counter))
    original_masks[original_masks!=0]=255
    secondary_pitd.putalpha(original_masks[key])
    secondary_pitd.save('difway_destroyed_image{}.png'.format(counter))
    """CREATE AN IMAGE WITH ONLY THE ALPHA CHANNEL DESTROYED"""
    pil_im_dest.putalpha(masks[key])
    tensor_im_total_dest = trans2(pil_im_total_dest)
    plt.imshow(tensor_im_total_dest[0])
    plt.show()
    # tensor_im_dest = trans2(pil_im_dest)
    # dest_tensor_list.append(tensor_im_dest)
    tensor_list.append(tensor_im)
    total_dest_tensor_list.append(tensor_im_total_dest)
    # filename = os.path.splitext(infile)[0].split("\\")[1]
    # im.save(dir_path3 + "/" + filename + ext)
    # os.remove(dir_path + "/" + filename + ext)
    counter += 1
    """STOP AT 10 IMAGES"""
    # if counter == 1000:
    data_counter += 1

data = torch.stack(tensor_list)
total_dest_data = torch.stack(total_dest_tensor_list)
torch.save(data, 'data{0}.pt'.format(data_counter))
torch.save(total_dest_data, 'total_dest_data{0}.pt'.format(data_counter))
    # total_dest_tensor_list = []
    # tensor_list = []
    # counter = 0

"""STACK THE DATA IN (NUMBER x CHANNELS x PIXELS x PIXELS)"""
"""ONLY THE ALPHA DESTROYED"""
# dest_data = torch.stack(dest_tensor_list)
"""GROUND TRUTH"""
# data = torch.stack(tensor_list)
# """TOTALLY DESTROYED"""
# total_dest_data = torch.stack(total_dest_tensor_list)
# """SAVE THE DATA IN PT FORMAT"""
# # torch.save(dest_data, 'dest_data.pt')
# torch.save(data, 'data.pt')
# torch.save(total_dest_data, 'total_dest_data.pt')
