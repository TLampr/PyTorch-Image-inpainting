from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import torchvision
import torch
from tqdm import tqdm

dir_path = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed/masks"
 #"/home/anala/data_crop_512_png"
# "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png"
dir_path2 = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed/"
#"/home/anala/data_crop_512_png/masks"
# "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/masks"
dir_path3 = "/home/anala/data_crop_512_png/Completed"#"C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png/Completed"
masks = {}
masks_tensors = {}
"""THE MASKS"""

# counter = 0
# for infile in tqdm(glob.glob(dir_path + "/*.png")):
#     file, ext = os.path.splitext(infile)
#     im = Image.open(infile)
#     im = im.resize((256,256),Image.ANTIALIAS)
#     trans = torchvision.transforms.ToPILImage()
#     trans2 = torchvision.transforms.ToTensor()
#     alpha_im = np.array(trans(trans2(im)).split()[3])
#     # alpha_im[alpha_im != 255] = 0
#     # alpha_im[alpha_im == 255] = 1
#     name = infile.split('/')[-1]
#     masks[name] = trans(trans2(alpha_im))
#     masks_tensors[name] = trans2(alpha_im)
#     counter+=1
# #     # im.save("{}.PNG".format(counter))
# #
# # torch.save(masks_tensors, 'masks.pt')
#
# key_array = np.array(list(masks.keys()))
# dest_tensor_list = []
tensor_list = []
total_dest_tensor_list = []
counter = 0
data_counter = 0
# new_counter = 0
masks = torch.load("masks.pt")
"""SAVES THE IMAGES AND THE DESTROYED ONES AS TWO BIG TENSORS"""
dir = glob.glob(dir_path2 + "/*.png")
ind = [1,1,9,0,4]

for infile, ind in [(dir[0],ind[0]),(dir[1],ind[1]),(dir[2],ind[2]),(dir[3],ind[3]),(dir[4],ind[4])]:
    # for infile in tqdm(glob.glob(dir_path2 + "/*.png")):
    file, ext = os.path.splitext(infile)
    """READS THE IMAGE"""
    im = Image.open(infile)
    im = im.resize((256, 256), Image.ANTIALIAS)
    """PIL TRANSFORMATION"""
    trans = torchvision.transforms.ToPILImage()
    """TENSOR TRANSFORMATION"""
    trans2 = torchvision.transforms.ToTensor()
    tensor_im = trans2(im)
    pil_im = trans(trans2(im))
    pil_im_dest = trans(trans2(im))
    """GET A RANDOM DESTRUCTION PATTERN"""
    # key = np.random.choice(key_array)
    key_list = list(masks.keys())
    mask = np.array(masks[key_list[ind]])
    """MAKE IT 1/0"""
    mask[mask != 0] = 1
    mask[mask == 0] = 0
    mask = np.transpose(mask,(1,2,0))

    """CREATE AN IMAGE WITH ALL RGB CHANNELS AFFECTED"""
    # pil_im_total_dest = trans(np.multiply(np.array(pil_im_dest), mask))#[:, :, None]))
    pil_im_total_dest = trans(trans2(np.multiply(np.array(pil_im_dest), mask)))  # [:, :, None]))
    mask = trans2(trans(mask))
    mask = np.array(mask[0,:,:])
    pil_im_total_dest.putalpha(mask)
    # new_counter+=1
        # pil_im_total_dest.save("d{}.png".format(new_counter))
        # """CREATE AN IMAGE WITH ONLY THE ALPHA CHANNEL DESTROYED"""
        # pil_im_dest.putalpha(masks[key])
    tensor_im_total_dest = trans2(pil_im_total_dest)
        # # tensor_im_dest = trans2(pil_im_dest)
        # # dest_tensor_list.append(tensor_im_dest)
        # tensor_list.append(tensor_im)
    total_dest_tensor_list.append(tensor_im_total_dest)
        # filename = os.path.splitext(infile)[0].split("\\")[1]
        # # im.save(dir_path3 + "/" + filename + ext)
        # # os.remove(dir_path + "/" + filename + ext)
        # counter += 1
        # """STOP AT 10 IMAGES"""
    if counter == 5:
        data = torch.stack(tensor_list)
        total_dest_data = torch.stack(total_dest_tensor_list)
        data_counter += 1
        torch.save(data, 'data.pt')
        torch.save(total_dest_data, 'total_dest_data.pt')
        #     total_dest_tensor_list = []
        #     tensor_list = []
        #     counter = 0
        # new_counter+=1

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
