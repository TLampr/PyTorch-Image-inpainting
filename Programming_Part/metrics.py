from skimage import measure

import numpy as np

def PSNR(y_true, y_pred):
  MSE = np.mean(np.square(y_pred - y_true), axis = (1,2,3))
  MAX_I = 1 #np.max(y_true) #max value of pixels
  PSNR = 20 * np.log10(MAX_I) - 10 * np.log10(MSE)  
  return PSNR

def l1(y_true, y_pred):
  l1 = np.mean(np.abs(y_pred - y_true), axis=(1,2,3))
  return l1

def compute_MSSIM(y_true, y_pred, window_size = 7) :
  MSSIM = np.zeros(y_true.shape[0])
  for i in range(y_true.shape[0]):
    for j in range(y_true.shape[1]) :
      MSSIM[i] += measure.compare_ssim(y_true[i,j,:,:], y_pred[i,j,:,:], win_size = window_size)
    MSSIM[i] /= y_true.shape[1] #average over channels
  return MSSIM