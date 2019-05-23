# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:32:51 2019

@author: depre
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from metrics import l1, PSNR, compute_MSSIM
from archi import UNet

if __name__ == '__main__':
    model = UNet(256)
    model.load_state_dict(torch.load('final/short_name_checkpoint20.pt'))
    model.eval()
    
    # Print model's state_dict
    """
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])
    """
    
    labels = torch.load("./data/data13.pt")
    data = torch.load("./data/total_dest_data13.pt")
        
    masks = data[:, 3, :, :][:, None, :, :]
    masks[masks != 0] = 1
    masks = torch.cat((masks, masks, masks), dim=1)
        
    with torch.no_grad():
        outputs = []
        masks_out = []
        
        avg_l1 = np.zeros(labels.shape[0])
        avg_PSNR = np.zeros(labels.shape[0])
        avg_MSSIM = np.zeros(labels.shape[0])
        
        for i in range(labels.shape[0]) :
            output = model(data[i, :3, :, :][None, :, :, :], masks[i, :, : ,:][None, :, :, :])
            
            l1_error = l1(output[0].numpy(), labels[i, :, :, :][None, :, :, :].numpy())
            PSNR_error = PSNR(output[0].numpy(), labels[i, :, :, :][None, :, :, :].numpy())
            MSSIM_error = compute_MSSIM(output[0].numpy(), labels[i, :, :, :][None, :, :, :].numpy())

            print('l1 error of the model : {}'.format(l1_error))
            print('PSNR error of the model : {}'.format(PSNR_error))
            print('MSSIM error of the model : {}'.format(MSSIM_error))
            
            des_im = data[:, :3, :, :][i].cpu().numpy()
            des_im = np.transpose(des_im, (1, 2, 0))
            plt.imshow(des_im)
            plt.axis('off')
            plt.show()
            
            matplotlib.image.imsave('image_destroyed{}.png'.format(i), des_im)
            
            im = output[0][0].cpu().numpy()
            im = np.transpose(im, (1, 2, 0))
            im -= im.min()
            im /= im.max()
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            
            matplotlib.image.imsave('image_reconstructed{}.png'.format(i), im)

            avg_l1[i] = l1_error
            avg_PSNR[i] = PSNR_error
            avg_MSSIM[i] = MSSIM_error

            outputs.append(output[0])
            masks_out.append(output[1])
            
        print('mean l1 error of the model : {}'.format(np.mean(avg_l1)))
        print('std l1 error of the model : {}'.format(np.std(avg_l1)))
        print('mean PSNR error of the model : {}'.format(np.mean(avg_PSNR)))
        print('std PSNR error of the model : {}'.format(np.std(avg_PSNR)))
        print('mean MSSIM error of the model : {}'.format(np.mean(avg_MSSIM)))
        print('std MSSIM error of the model : {}'.format(np.std(avg_MSSIM)))




