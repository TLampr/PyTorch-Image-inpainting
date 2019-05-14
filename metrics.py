from sklearn import metrics
from skimage import measure

import time



def PSNR(y_true, y_pred):
  MSE = np.mean(np.square(y_pred - y_true), axis = (1,2,3))
  MAX_I = np.max(y_true) #max value of pixels
  PSNR = 20 * np.log10(MAX_I) - 10 * np.log10(MSE)  
  return PSNR

def l1(y_true, y_pred):
  l1 = np.mean(np.abs(y_pred - y_true), axis=(1,2,3))
  return l1

def compute_SSIM(x, y):
  #x, y windows of y_true and y_pred of common size NxN 
  k1 = 0.01
  k2 = 0.03
  L = 1
  #L the dynamic range of the pixel-values (typically this is 2^#bits_per_pixel-1)
  #L = 255 for 8-bit grayscale images
  c1 = (k1*L)**2
  c2 = (k2*L)**2
  
  SSIM = np.zeros((x.shape[0],x.shape[1]))
  
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      covxy = np.mean(x[i,j,:,:] - np.mean(x[i,j,:,:], axis = (0,1)))*np.mean(y[i,j,:,:] - np.mean(y[i,j,:,:], axis = (0,1)))#np.cov(x[i,j,:,:], y[i,j,:,:])
      covxx = np.mean(np.square(x[i,j,:,:] - np.mean(x[i,j,:,:], axis = (0,1)))) #np.cov(x[i,j,:,:], x[i,j,:,:])
      covyy = np.mean(np.square(y[i,j,:,:] - np.mean(y[i,j,:,:], axis = (0,1)))) #np.cov(y[i,j,:,:], y[i,j,:,:])
      SSIM[i,j] = (2*np.mean(x[i,j,:,:], axis = (0,1))*np.mean(y[i,j,:,:], axis = (0,1)) + c1)*(2*covxy + c2)/((np.mean(x[i,j,:,:], axis = (0,1))**2 + np.mean(y[i,j,:,:], axis = (0,1))**2 + c1)*(covxx**2 + covyy**2 + c2))
  return SSIM

def MSSIM(y_true, y_pred):
  window_size = (7,7)
  SSIM = np.zeros((y_true.shape[0], y_true.shape[1], int(np.ceil(y_true.shape[2]/window_size[0])), int(np.ceil(y_true.shape[3]/window_size[1]))))
  print(SSIM.shape)
  for i in range(SSIM.shape[2]) :
    for j in range(SSIM.shape[3]):
      SSIM[:,:,i,j] = compute_SSIM(y_true[:,:, i*window_size[0] : (i+1)*window_size[0], j*window_size[1] : (j+1)*window_size[1]],
                                   y_pred[:,:, i*window_size[0] : (i+1)*window_size[0], j*window_size[1] : (j+1)*window_size[1]])
  print(SSIM)
  MSSIM = np.mean(SSIM, axis = (1,2,3))
  return MSSIM
  
y_true = torch.rand(10, 3, 512, 512)
y_pred = torch.rand(10, 3, 512, 512)

y_true = y_true.numpy()
y_pred = y_pred.numpy()

print("PSNR : ", PSNR(y_true, y_pred))
print("l1 : ", l1(y_true, y_pred))

start = time.time()
print("MSSIM : ", MSSIM(y_true, y_pred))
fin = time.time()
print('duree : ', fin - start)

def compute_MSSIM(y_true, y_pred, window_size) :
  MSSIM = np.zeros(y_true.shape[0])
  for i in range(y_true.shape[0]):
    for j in range(y_true.shape[1]) :
      MSSIM[i] += measure.compare_ssim(y_true[i,j,:,:], y_pred[i,j,:,:], win_size = window_size)
    MSSIM[i] /= y_true.shape[1] #average over channels
  return MSSIM

window_size = 7
print("MSSIM average over nb of channels :", compute_MSSIM(y_true, y_pred, window_size))
