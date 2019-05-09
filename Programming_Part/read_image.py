from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import torchvision
import tqdm

dir_path = "C:/Users/anala/Desktop/coursecontent/Deep Learning/Project/Our-Lovely-Awesome-Team-Project/CelebA/img_celeba/data_crop_512_png"
size = 512, 512
counter = 1
# for filename in os.listdir(dir_path):
#     if filename.endswith(".jpg"):
#         im = Image.open(dir_path + "/" + filename)
#         filename = filename.split(".")[0]
#         filename = filename + ".png"
#         im.save(filename)
#         counter +=1
#         print(counter)
#     if counter > 50:
#         break

for infile in glob.glob(dir_path + "/*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    trans = torchvision.transforms.ToPILImage()

    trans2 = torchvision.transforms.ToTensor()
    plt.imshow(trans(trans2(im)))
    tensor_im = trans2(im)
    pil_im = trans(trans2(im))
    alpha_im = np.array(trans(trans2(im)).split()[3])
    alpha_im[alpha_im != 255] = 0
    destroyed_im = np.array(pil_im)
    destroyed_im[:,:,3] = alpha_im
    print('hi')


print('hi')


