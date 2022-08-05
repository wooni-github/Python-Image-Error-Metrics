# https://github.com/wooni-github/Python-Image-Error-Metrics/edit/main/ImageErrorMetric.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import numpy as np
from skimage import metrics
import cv2
import math
import datetime
from torch import nn
import torch
import torch_SSIM
import pytorch_mssim

_round = 3

def RMSE_numpy(GT, Img, isMSE = False):
    '''
    Must convert Img (uint8) to float/double to calculate correct results.
    ex) uint8 : a (0), b (100)
    ==> a-b = 0 - 100 = 256-100 = 156 (incorrect)
    ==> b-a = 100 - 0 = 100 (correct)
    '''
    GT_ = np.array(GT, dtype=np.float32)
    Img_ = np.array(Img, dtype=np.float32)
    MSE = np.mean((GT_-Img_)**2)
    
    if isMSE:
        return round(MSE, _round)
    
    return round(np.sqrt(MSE), _round)

def RMSE_ForLoop(GT, Img, isMSE = False):
    GT_ = np.array(GT, dtype=np.float32)
    Img_ = np.array(Img, dtype=np.float32)
    h, w, c = GT.shape
    sum = 0
    for y in range(h):
        for x in range(w):
            for d in range(c):
                sum +=  (GT_[y, x, d] - Img_[y, x, d])**2
    sum /= (h*w*c)
    
    if isMSE:
        return round(sum, _round)
    
    return round(np.sqrt(sum), _round)

def RMSE_ForLoop_ROI(GT, Img, Mask, isMSE = False):
    GT_ = np.array(GT, dtype=np.float32)
    Img_ = np.array(Img, dtype=np.float32)
    h, w, c = GT.shape
    sum = 0
    n = 0
    for y in range(h):
        for x in range(w):
            if Mask[y, x, 0] == 255 and Mask[y, x, 1] == 255 and Mask[y, x, 2] == 255:
                for d in range(c):
                    sum += (GT_[y, x, d] - Img_[y, x, d])**2
                    n += 1
    sum /= n  
        
    if isMSE:
        return round(sum, _round)
    
    return round(np.sqrt(sum), _round)

def RMSE_numpy_Bit_ROI(GT, Img, Mask, isMSE = False):
    masked_GT = cv2.bitwise_and(GT, Mask)  # GT의 마스크 이미지영역만 crop
    masked_Img = cv2.bitwise_and(Img, Mask)

    masked_GT = np.array(masked_GT, dtype=np.float32)
    masked_Img = np.array(masked_Img, dtype=np.float32)

    n = np.count_nonzero(Mask)
    sum = ((masked_GT - masked_Img) ** 2).sum()
    sum /= n
        
    if isMSE:
        return round(sum, _round)
    
    RMSE = round(np.sqrt(sum), _round)
    return RMSE

def SSIM_skimage(GT, Img):
    '''window_size, C1, C2, sigma from torch_SSIM.py'''
    window_size = 11
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    sigma = 1.5

    # grayA = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    # SSIM = metrics.structural_similarity(grayA, grayB, win_size=window_size, K1 = C1, K2 = C2, multichannel=False, gaussian_weights=True, sigma=sigma, use_sample_covariance=True, data_range=255)

    SSIM = metrics.structural_similarity(GT, Img, win_size=window_size, K1=C1, K2=C2, multichannel=True, gaussian_weights=True, sigma=sigma, use_sample_covariance=True, data_range=255)

    return round(SSIM, _round)

def PNSR(GT, Img, MSEMethod):
    MSE = MSEMethod(GT, Img, isMSE = True)
    if MSE == 0:
        return 100
    PIXEL_MAX = 255.0
    return round(20 * math.log10(PIXEL_MAX / math.sqrt(MSE)), _round)


def preprocess(img):
    x = np.array(img).astype(np.float32)
    x = torch.from_numpy(x)
    return x

def RMSE_torch(GT, Img):
    criterion = nn.MSELoss()
    GT_ = preprocess(GT)
    Img_ = preprocess(Img)
    MSE = criterion(GT_, Img_)
    return round(np.sqrt(MSE.cpu().detach().numpy()), _round)

def RMSE_torch(GT, Img):
    criterion = nn.MSELoss()
    GT_ = preprocess(GT)
    Img_ = preprocess(Img)
    MSE = criterion(GT_, Img_)
    return round(np.sqrt(MSE.cpu().detach().numpy()), _round)

def SSIM_torch(GT, Img):
    # https://github.com/Po-Hsun-Su/pytorch-ssim

    # grayA = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    # GT_ = torch.unsqueeze(torch.unsqueeze(preprocess(grayA), 0), 0)
    # Img_ = torch.unsqueeze(torch.unsqueeze(preprocess(grayB), 0), 0)
    # SSIM = torch_SSIM.ssim(GT_, Img_).numpy()

    GT_ = torch.unsqueeze(preprocess(GT), 0).permute((0, 3, 1, 2))
    Img_ = torch.unsqueeze(preprocess(Img), 0).permute((0, 3, 1, 2)) # [b, c, h, w]
    SSIM = torch_SSIM.ssim(GT_, Img_).numpy()
    # window_size = 11,
    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2
    # sigma = 1.5

    return np.round(SSIM, _round)

def SSIM_torch2(GT, Img):
    # https://github.com/VainF/pytorch-msssim
    GT_ = torch.unsqueeze(preprocess(GT), 0).permute((0, 3, 1, 2))
    Img_ = torch.unsqueeze(preprocess(Img), 0).permute((0, 3, 1, 2))
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)

    ssim_val = round(pytorch_mssim.ssim( GT_, Img_, data_range=255, size_average=False, K = (0.01 ** 2, 0.03 ** 2), nonnegative_ssim=True).detach().numpy()[0], 3)
    return ssim_val

def RMSE_skimage(GT, Img, isMSE = False):
    if isMSE:
        return metrics.mean_squared_error(GT, Img)
    return round(math.sqrt(metrics.mean_squared_error(GT, Img)), _round)

if __name__ == '__main__':
    '''
    1) GT, Input, Mask images have same width and height
    2) Reference RMSE : Pytorch nn.MSELoss 
    3) Assume all images have at least one maximum RGB value, 255 (for PNSR)
    4) SSIM is calculated using parameters of torch_SSIM.py 
    5) for RMSE in ROI, Mask image is provided in binary image. Black(0) : RONI(Region of Non-Interest), White(255) : ROI(Region of Interest)   
    '''
    GT = cv2.imread('TestImage1.png')
    Input = cv2.imread('TestImage2.png')


    print('**** Structural Similarity Index (SSIM)')
    print('[Torch1]', SSIM_torch(GT, Input))
    print('[Torch2]', SSIM_torch2(GT, Input))
    print('[skimage]', SSIM_skimage(GT, Input))
    print()

    d = datetime.datetime.now()
    print('**** Root mean square error (RMSE)')
    print('[numpy] ', RMSE_numpy(GT, Input), '// calculation time : ', (datetime.datetime.now()-d).microseconds/10e5, 'sec'); d = datetime.datetime.now()
    print('[skimage] ', RMSE_skimage(GT, Input), '// calculation time : ', (datetime.datetime.now()-d).microseconds/10e5, 'sec'); d = datetime.datetime.now()
    print('[Torch]', RMSE_torch(GT, Input), '// calculation time : ', (datetime.datetime.now()-d).microseconds/10e5, 'sec'); d = datetime.datetime.now()
    print('[ForLoop]', RMSE_ForLoop(GT, Input), '// calculation time : ', (datetime.datetime.now()-d).seconds, 'sec'); d = datetime.datetime.now()

    print()
    print('**** Peak signal-to-noise ratio (PSNR)')
    print(PNSR(GT, Input, RMSE_numpy), 'dB')

    print()
    print('**** Root mean square error (RMSE) for Region of Interest(ROI)')
    Mask = cv2.imread('MaskImage.png')
    d = datetime.datetime.now()
    print('[Numpy + Bit]', RMSE_numpy_Bit_ROI(GT, Input, Mask), '// calculation time : ', (datetime.datetime.now()-d).microseconds/10e5, 'sec'); d = datetime.datetime.now()
    print('[ForLoop]', RMSE_ForLoop_ROI(GT, Input, Mask),  '// calculation time : ', (datetime.datetime.now()-d).seconds, 'sec'); d = datetime.datetime.now()
