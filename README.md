# Python Image Error Metrics
Python image error/similarity calculation examples (RMSE, PSNR, SSIM, RMSE for ROI) using OpenCV, numpy, skimage, pytorch

Image error metrics between two images TestImage1.png and TestImage2.png

|Image1|Image2|MaskImage|
|:---:|:---:|:---:|
|![TestImage1](https://user-images.githubusercontent.com/84174755/182813457-d810d6f1-3ecf-4ba9-9885-be22b9807d14.png)|![TestImage2](https://user-images.githubusercontent.com/84174755/182813948-4bd81720-328c-41ca-ab4f-5a1e91925dff.png)|![MaskImage](https://user-images.githubusercontent.com/84174755/182813961-8ae02ab6-a0b1-4bb0-b841-294535053b16.png)|


파이썬을 이용한 이미지간의 에러/유사도를 비교하는 예제입니다.
+ RMSE (MSE)
+ PSNR
+ SSIM
+ RMSE (ROI) : Mask이미지는 검정(배경), 흰색(관심영역)으로 제공. 관심영역에 대해서만 RMSE를 계산

4가지를 표기하며 기본적으로 OpenCV

+ numpy
+ skimage
+ pytorch

를 이용합니다.

근본적인 목적은 Pytorch에서 CNN을 이용한 학습을 수행할 때 사용한 MSELoss나 SSIMloss 등이 어떻게 계산되는지 파악하기 위함입니다.

Environment : 

    conda create -n PyImgErr python==3.8
    conda activate PyImgErr
    conda install numpy
    conda install scikit-image
    conda install opencv-python
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

Or :

    conda env create --file environment.yaml
    conda activate PyImgErr
    
Run : 

    python ImageErrorMetric.py
    

# **Structural Similarity Index (SSIM)**

SSIM은 CompenNet (https://github.com/BingyaoHuang/CompenNet) 에서 Training loss 계산시에 사용하는 파라미터를 기준으로 계산합니다.

Parameters (same as Torch1)
+ window_size = 11
+ C1 = 0.01 ** 2
+ C2 = 0.03 ** 2
+ sigma = 1.5

|Method|SSIM||
|:---:|:---:|:---|
|[Torch1]|0.426|Po-Hsun Su (https://github.com/Po-Hsun-Su/pytorch-ssim)|
|[Torch2]|0.428|Gongfan Fang (https://github.com/VainF/pytorch-msssim)|
|[skimage]|0.428||

각 방식별 약간의 차이는 있지만 큰 차이가 없는 결과를 보여줍니다.

# **Root mean square error (RMSE)**

|Method|RMSE|Calculation time|
|:---:|:---:|:---:|
|[numpy]|68.871|0.054946 sec|
|[skimage]|68.871|0.047014 sec|
|[Torch]|68.871|0.019004 sec|
|[ForLoop]|68.871|9 sec|

이미지 - 이미지 간에 RMSE를 계산하는 방법에는 두가지가 있습니다.

전체 n개의 픽셀에 대해 


$i = {1, 2, ... n}$


이미지1과 이미지2에서 각각 픽셀의 RGB가 이와 같이 주어지면


$RGB1_{i} =(R1_{i}, G1_{i}, B1_{i})$ 

$RGB2_{i} =(R2_{i}, G2_{i}, B2_{i})$

$RMSE1 = \sqrt{MSE} = \sqrt{\sum\limits_{i=1}^n \frac{(R1_i-R2_i)^{2}+(G1_i-G2_i)^{2}+(B1_i-B2_i)^{2}}{n}}$


로 계산하는 것이 일반적입니다. 3D 벡터형태로 주어지는 n개의 RGB에 대한 평균오차의 루트값이 RMSE이기 때문이죠.

* * *

다만, 일부 연구에서는 아래와 같은 식을 사용하기도 합니다. n개의 픽셀이 각각 3채널의 값을 갖고 있기 때문에 분모에 n이 아닌 3n을 사용하는 경우입니다.


$RMSE2 = \sqrt{MSE} = \sqrt{\sum\limits_{i=1}^n \frac{(R1_i-R2_i)^{2}+(G1_i-G2_i)^{2}+(B1_i-B2_i)^{2}}{3n}}$


PyTorch의 nn.MSELoss()를 사용하는 경우에는 $RMSE2$ 와 같은 값을 갖게 됩니다.

사실 RMSE1과 RMSE2 사이에는 $\sqrt{3}\$배의 scale 차이 밖에 없어 아주 큰 의미를 갖는 내용은 아니지만, 논문에서 서술하는 오차값이 실제 구현했을때 오차값과 차이가 나는 경우가 이런 경우에 해당할 때가 많더라구요.

* * *

한 가지 더 주목할 점은, Triple for loop (width, height, channel) 로 모든 픽셀에 접근하여 RMSE를 직접 계산하는 방식은 **매우 비효율적**이라는 것입니다.

예제 이미지는 [1920 x 1080] FHD 이미지인데, 이를 다른 3가지 방법으로 계산하는 경우에는 매우 빠른 시간 내에 계산이 가능하지만, 직접 계산하는 경우에는 9초라는 무시하지 못할 시간이 걸립니다. GPU를 사용하는 Pytorch는 그렇다 치더라도, numpy에 비해 엄청나게 느린 속도죠.

이는 numpy가 C로 만든 모듈이고, 내부적으로 **C를 원시타입으로 변환하여 계산하기 때문에 일반적인 파이썬 계산 방법에 비해 매우 빠른 계산 속도를 제공**하기 때문입니다.

또한, 직접 픽셀에 접근하여 오차를 계산하고자 하는 경우 주의할 점이, CV/PIL 등으로 로드한 이미지는 파이썬에서 **8비트로** 데이터를 관리합니다.

0~255로 표현되는 수치에서 오차를 계산하게 되면 $|a-b|$ 와 $|b-a|$ 가 서로 다른 값을 갖을 수 있습니다. **float으로 변경하여** 계산해줍니다.



# **Peak signal-to-noise ratio (PSNR)**

+ 11.37 dB

PSNR은 아래 수식으로 계산됩니다. $MAX_{I}$ 는 이미지에서 가장 큰 값을 갖는 수치인데 보통 255로 설정합니다.

$PSNR = 10log_{10}{( \frac{MAX_{I}^{2}}{MSE})} = 20log_{10}{( \frac{MAX_{I}}{RMSE})}$


# **Root mean square error (RMSE) for Region of Interest(ROI)**
|Method|RMSE (ROI)|Calculation time|
|:---:|:---:|:---:|
|[Numpy + Bit]|69.486|0.029008 sec|
|[ForLoop]|69.486|5 sec|
 
관심영역에 대한 RMSE계산입니다. 앞서 언급한 대로, Triple For Loop를 사용하는 방법은 매우 비효율적입니다.

심지어 ROI 영역에 대한 계산만을 수행할 때, Mask이미지가 True 인지(흰색) False(검은색) 인지 구분하는 과정까지 포함하면 더욱 더 기피해야할 방법이죠.

더 깔끔하게 계산할 수 있는 분들이 있으실텐데, 저는 이런 방식을 택했습니다.

|Masked1|Masked2|
|:---:|:---:|
|![BitGT](https://user-images.githubusercontent.com/84174755/182814587-f268f0d2-6650-4ae1-94c4-a58fdf4f6e83.png)|![BitImg](https://user-images.githubusercontent.com/84174755/182814591-7efb8286-cc8d-432e-be80-d5d7f7c915e2.png)|

1. Img1, Img2를 Mask 이미지와 함께 Bit-wise 연산하여 masked 이미지 생성
2. np.count_nonzero(Mask) 를 이용한 non-zero 픽셀 갯수 계산
3. 기존 RMSE방식과 유사하게 Masked1, Masked2 간의 전체 픽셀 차이의 합 계산
4. 계산한 오차를 non-zero 픽셀 수로 나눔





