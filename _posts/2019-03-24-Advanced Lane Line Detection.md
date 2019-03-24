
# 레인 디텍션 openCV

우리는 학습할 데이터가 없기 때문에 딥러닝보단 고전적인 레인 디텍션을 써야하는 환경 

과정

1. 카메라로 사진 읽기 
2. 왜곡 조정
3. Gradient 와 Color 필터링   
4. 시야 조정
5. 라인 서치 및 서치 최적화
6. 선 덧 입혀서 그리기


```python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
import io
import os
import glob
%matplotlib inline
```


```python
pwd!
```




    '/home/pirl/Downloads/edu_/Lane Detection_hoon'




```python
# 이미지 읽기
images = glob.glob('./camera_cal/calibration*.jpg')
# 이미지 보여주기
img = mpimg.imread(images[0])
plt.imshow(img);
# mpimg.imread() 로 이미지 읽으면, cv2.COLOR_RGB2GRAY 로 회색조 음영
# cv2.imread() 로 이미지 읽으면, cv2.COLOR_BGR2GRAY 로 회색조하기
```


![png](https://i.imgur.com/V1sXxDu.png)


# 사진 읽기

체스 판에 대상 지점 수를 저장(바깥 쪽 가장자리에없는 점만 고려)
모든 z 값은 2D 이미지이므로 0


```python
# 체스의 좌표축 저장
chess_points = []
# 체스는 6 rows , 9 columns z는 0 
chess_point = np.zeros((9*6, 3), np.float32)
chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
# 왜곡 보정 체스의 좌표축 저장
image_points = []
```


```python
for image in images:
    img = mpimg.imread(image)
    # 회색조로 만들기
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # returns boolean and coordinates
    success, corners = cv.findChessboardCorners(gray, (9,6), None)
    if success:
        image_points.append(corners)
        # these will all be the same since it's the same board
        chess_points.append(chess_point)
    else:
        print('corners not found {}'.format(image))
```

    corners not found ./camera_cal/calibration5.jpg
    corners not found ./camera_cal/calibration4.jpg
    corners not found ./camera_cal/calibration1.jpg



```python
image = mpimg.imread('./camera_cal/calibration3.jpg')

plt.figure();
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10));
ax1.imshow(image);
ax1.set_title('원본이미지', fontsize=30);

gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) 
ret , corners = cv.findChessboardCorners(gray,(9,6),None)    
if ret == False:
    print('cann''t find corner')
img1 = cv.drawChessboardCorners(image,(9,6),corners,ret) 

ax2.imshow(img1);
ax2.set_title('보정전처리이미지', fontsize=30);
plt.tight_layout();
plt.show;
```


    <Figure size 432x288 with 0 Axes>



![png](https://i.imgur.com/yufZIba.png)


사진의 모서리를 성공적으로 mapping

아직 남은 문제가 빛의 굴곡, 사실은 체스판이 일자로 되어 있어야하는데 사진을 저렇게 찍어서 휘어져 있음.  
오른쪽 두번째 제일 위에 파란색 길이와 빨간색 길이가 같아야함 
왜곡 문제가 있음  

### 왜곡보정


```python
points_pickle = pickle.load(open( "object_and_image_points.pkl", "rb" ) )
chess_points = points_pickle["chesspoints"]
image_points = points_pickle["imagepoints"]
img_size = points_pickle["imagesize"]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)
#포인트를 이진으로 저장하기
```


```python
def distort_correct(img,mtx,dist,camera_img_size):
    img_size1 = (img.shape[1],img.shape[0])
    print(img_size1)
    print(camera_img_size)
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist

img = mpimg.imread('./camera_cal/calibration3.jpg')
img_size1 = (img.shape[1], img.shape[0])

undist = distort_correct(img, mtx, dist, img_size)

# 이미지 출력
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10));
ax1.imshow(img);
ax1.set_title('원본이미지', fontsize=30);
ax2.imshow(undist);
ax2.set_title('왜곡보정이미지', fontsize=30);
plt.tight_layout()
plt.savefig('saved_figures/undistorted_chess.png')
```

    (1280, 720)
    (1280, 720)



    <Figure size 432x288 with 0 Axes>



![png](https://i.imgur.com/sIGdubo.png)


젤 윗 부분을 보면 확연히 달라짐을 볼 수 있다.  
휘어진 네모들이 일자로 펼쳐져 있다.  

# Gradient & Color Thresholding
차선을 감지하기 위해, 여러가지 필터를 사용해서 흑백 사진을 만든다
* Sobel gradients in the x & y directions
* Gradient magnitude
* Gradient direction
* Color space transform and filtering


```python
# 카메라와 왜곡 행렬생성
camera = pickle.load(open( "camera_matrix.pkl", "rb" ))
mtx = camera['mtx']
dist = camera['dist']
camera_img_size = camera['imagesize']

# 이미지 왜곡받고 왜곡 처리
image = mpimg.imread('test_images/test1.jpg')
image = distort_correct(image,mtx,dist,camera_img_size)
plt.imshow(image);
```

    (1280, 720)
    (1280, 720)



![png](https://i.imgur.com/WvdCeBG.png)


# Sobel gradients


```python
def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # 회색조 만들기
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # x 또는 y에 OpenCV Sobel() function적용 
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # 리스케일
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

# 왜곡보정후 RGB의 파일을 흑백으로
plt.imshow(abs_sobel_thresh(image, thresh=(20,110)),  cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f19e865e1d0>




![png](https://i.imgur.com/4Z09eQb.png)


# Gradient Magnitude

그레디언트의 최소 / 최대 크기를 기준으로 필터링, 이 함수는 단일 방향의 크기 또는 두 가지 선형 조합을 필터링로도 가능


```python
def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 회색조 만들기
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # x와 y 각각의 sobel gradient
    x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # mag를 0,0 과 x,y 사이의 거리로 정의
    mag = np.sqrt(x**2 + y**2)
    # 리스케일
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    # mag를 binary mask로
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1 
    return binary_output

plt.imshow(mag_threshold(image, thresh=(20,100)),  cmap='gray');
```


![png](https://i.imgur.com/HiVV7CI.png)


# Gradient Direction
그레디언트의 방향을 기준으로 필터링 , lane detection을 위해선 90도 부근의 +/-에 관심을 가져야 한다


```python
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 회색조 만들기
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # x와 y 각각의 sobel gradient
    x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel))
    # 그래디언트의 방향을 계산하기 위해 np.arctan2(abs_sobely, abs_sobelx)사용 
    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output

plt.imshow(dir_threshold(image, thresh=(0.8,1.2)),  cmap='gray');
```


![png](https://i.imgur.com/uu6NrON.png)


# 채도 및 적색 필터

이제까지의 그래디언트 필터는 원본을 흑백으로 변환하기 때문에 많은 정보가 손실되었다. 이런이유로 노란색 또는 흰색의 차선을 찾기 어려웠고 우리는 색,채도,밝기 색상 공간을 이용할것이다. 특히 이런 방식은 차선에 대해 많은 정보를 알 수 있다. 특히 적색 필터는 도로에 그림자가 있을경우, 그나마 가장 잘 도로를 인식 할 수 있다.


```python
def hls_select(img, sthresh=(0, 255),lthresh=()):
    # 색 채도 밝기 맵으로 바꾸기
    hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # S 채널에 threshold 적용
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 이 결과를 토대로 바이너리 맵 
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1]) & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output
plt.imshow(hls_select(image, sthresh=(140,255), lthresh=(120, 255)));
```


![png](https://i.imgur.com/aAP1o1W.png)



```python
def red_select(img, thresh=(0, 255)):
    # 적색필터 적용
    R = img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output
plt.imshow(red_select(image, thresh=(200,255)));
```


![png](https://i.imgur.com/5TVBAws.png)


# 여러 필터 방법 합치기


```python
def binary_pipeline(img):
    img_copy = cv.GaussianBlur(img, (3, 3), 0)
    #img_copy = np.copy(img)
    
    # color channels
    s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
    #red_binary = red_select(img_copy, thresh=(200,255))
    
    # Sobel x
    x_binary = abs_sobel_thresh(img_copy,thresh=(25, 200))
    y_binary = abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
    xy = cv.bitwise_and(x_binary, y_binary)
    
    #magnitude & direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))
    
    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv.bitwise_or(s_binary, gradient)
    
    return final_binary

result = binary_pipeline(image)

# 결과출력
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result, cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.tight_layout()
plt.savefig('saved_figures/combined_filters.png')
```


    <Figure size 432x288 with 0 Axes>



![png](https://i.imgur.com/wmBSMem.png)



```python
# 왜곡보정
image = mpimg.imread('test_images/test5.jpg')
image = distort_correct(image,mtx,dist,camera_img_size)
result = binary_pipeline(image)

# 필터적용
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result, cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

    (1280, 720)
    (1280, 720)



![png](https://i.imgur.com/WDsRpMx.png)


# 시야 변경

3D를 2D로 바꿔준다 생각 하면 됨  
사람 눈엔 2D이미지를 봐도 3d이미지로 생각하게 됨  
컴퓨터는 그렇지 않음  
우리가 생각하는 3D를 2D로 만들어줘서 컴퓨터가 알 수 있게 만들어야함  
아래 그림 참고 


```python
def warp_image(img):
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]

    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    source_points = np.float32([ [0.117 * x, y] , [(0.5 * x) - (x*0.078), (2/3)*y] , 
                                [(0.5 * x) + (x*0.078), (2/3)*y] , [x - (0.117 * x), y] ])
    
    destination_points = np.float32([ [0.25 * x, y] , [0.25 * x, 0] , [x - (0.25 * x), 0] ,
                                     [x - (0.25 * x), y] ])
    
    perspective_transform = cv.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv.getPerspectiveTransform( destination_points, source_points)
    warped_img = cv.warpPerspective(img, perspective_transform, image_size, flags=cv.INTER_LINEAR)
    
    #print(source_points)
    #print(destination_points)
    
    return warped_img, inverse_perspective_transform
```


```python
birdseye_result, inverse_perspective_transform = warp_image(result)

plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

image_size = (image.shape[1], image.shape[0])
x = image.shape[1]
y = image.shape[0]
source_points = np.int32([ [0.117 * x, y] , [(0.5 * x) - (x*0.078), (2/3)*y],
                          [(0.5 * x) + (x*0.078), (2/3)*y] , [x - (0.117 * x), y] ])
draw_poly = cv.polylines(image,[source_points],True,(255,0,0), 5)

ax1.imshow(draw_poly)
ax1.set_title('Source', fontsize=40)
ax2.imshow(birdseye_result, cmap='gray')
ax2.set_title('Destination', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.tight_layout()
plt.savefig('saved_figures/perspective_transform.png')
```


    <Figure size 432x288 with 0 Axes>



![png](https://i.imgur.com/JUJdxtW.png)



```python
# 다시 기본이미지 저장
image = mpimg.imread('test_images/test5.jpg')
image = distort_correct(image,mtx,dist,camera_img_size)
```

    (1280, 720)
    (1280, 720)


# Lane Detection


```python
# 위의 Destination 이미지에서 절반만 추출 그래프에서 빈도수가 높은 지역은 흰색(Lane) 부분이 많은 지역
histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):,:], axis=0)
plt.figure();
plt.plot(histogram)
```




    [<matplotlib.lines.Line2D at 0x7f19e833ae10>]




![png](https://i.imgur.com/Era12BI.png)


```python
def track_lanes_initialize(binary_warped):
    
    global window_search
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # 이미지를 그리고, 그것의 결과를 보여준다
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # 히스토그램 절반 각각의 최대값이 필요하다
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # 슬라이딩 윈도우의 갯수를 정하고, 만약 이미지의 높이를 정확하게 나누지 않는다면 에러가 발생한다.
    nwindows = 9
    # 창의 높이를 설정한다
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # 이미지의 모든 nonzero픽셀에서 x와 y위치를 확인한다
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 각 윈도우의 현재 위취를 업데이트 한다
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 창의 넓이의 margin을 정하고, 창 안에 있는 픽셀 최소 수를 정한다
    margin = 100
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit,right_fit
```


```python
left_fit,right_fit = track_lanes_initialize(birdseye_result)
Image('saved_figures/01_window_search.png')
```




![png](https://i.imgur.com/gJdDmwM.png)



저렇게 박스형태로 차선을 인식하는 것이 좋다.  
앙상블이 좋은 것처럼~

차선 별견 완료 -> 검색속도 증가 (검색 범위 조정으로)
1. 다항식을 가져오기
2. 픽셀 마진 내에서 0이 아닌 픽셀을 찾기
3. 새로운 데이터에 맞도록 다항식을 갱신


```python
def track_lanes_update(binary_warped, left_fit,right_fit):

    global window_search
    global frame_count
    
    # repeat window search to maintain stability
    if frame_count % 10 == 0:
        window_search=True
   
        
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    return left_fit,right_fit,leftx,lefty,rightx,righty
```


```python
global frame_count
frame_count=0
left_fit,right_fit,leftx,lefty,rightx,righty = track_lanes_update(birdseye_result, left_fit,right_fit)
Image('saved_figures/02_updated_search_window.png')
```




![png](https://i.imgur.com/zSwfJ8d.png)



# 원본에 레인 그리기
1. 다항식을 가져오기
2. 변형 된 이진 파일의 빈 복사본에 다항식 곡선을 그리기 
3. 곡선 사이에 다각형을 채 우기 
4. 역 투시 변환을 사용하여 새 이미지를 unwap
5. 차선에 lane iamge 덧 입히기


```python
def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

def lane_fill_poly(binary_warped,undist,left_fit,right_fit):
    
    # Generate x and y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_val(ploty,left_fit)
    right_fitx = get_val(ploty,right_fit)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane 
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp using inverse perspective transform
    newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
    # overlay
    #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
        
    return result

colored_lane = lane_fill_poly(birdseye_result, image, left_fit, right_fit)
plt.figure()
plt.imshow(colored_lane);
plt.tight_layout()
plt.savefig('saved_figures/lane_polygon.png')
```


![png](https://i.imgur.com/B5TG0ug.png)


# 라인 곡률 구하기

이제 차선이 생겼으므로 곡률 반경 (즉, 도로가 얼마나 휘어지고 있는지)을 계산하려고한다. 이 정보는 자동차의 조향 및 가속을 제어하는 프로그램을 작성해야 할 때 엔드 투 엔드 학습 프로세스에서 중요한 역할을 한다. 이 과정에서 가장 중요한 단계는 측정을 픽셀 공간에서 미터법으로 변환하는 것입니다. 곡률 반경 공식:

$$R_{curve}=\frac{[1+(\frac{dx}{dy})^2]^\frac{3}{2}}{|\frac{d^2x}{dy^2}|}$$

$$f(y)=Ay^2+By+C$$

$$f'(y)=\frac{dx}{dy}=2Ay+B$$

$$f''(y)=\frac{d^2x}{dy^2}=2A$$
커브 공식:
$R_{curve}=\frac{[1+(2Ay+B)^2]^\frac{3}{2}}{|2A|}$


```python
def measure_curve(binary_warped,left_fit,right_fit):
        
    # generate y values 
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # measure radius at the maximum y value, or bottom of the image
    # this is closest to the car 
    y_eval = np.max(ploty)
    
    # coversion rates for pixels to metric
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
   
    # x positions lanes
    leftx = get_val(ploty,left_fit)
    rightx = get_val(ploty,right_fit)

    # fit polynomials in metric 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # calculate radii in metric from radius of curvature formula
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # averaged radius of curvature of left and right in real world space
    # should represent approximately the center of the road
    curve_rad = round((left_curverad + right_curverad)/2)
    
    return curve_rad

measure_curve(birdseye_result,left_fit, right_fit)
```




    1338.0



# 차량이 차선 어디에 있는지 확인하기
자동차에 장착되어 있는 카메라가 정 중앙에 있다고 가정


```python
def vehicle_offset(img,left_fit,right_fit):
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 3.7/700 
    image_center = img.shape[1]/2
    
    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    right_low = get_val(img.shape[0],right_fit)
    
    # pixel coordinate for center of lane
    lane_center = (left_low+right_low)/2.0
    
    ## vehicle offset
    distance = image_center - lane_center
    
    ## convert to metric
    return (round(distance*xm_per_pix,5))

offset = vehicle_offset(colored_lane, left_fit, right_fit)
print(offset)
```

    -0.00641


## 이미지 처리
이젠 모든 처리를 구축했다. 이제 비디오 영상에서 실행되도록 모든 처리를 결합해야한다. 
방식은 처리 클래스를 만들고, 다른 속성을 계속해서 추적하게 만드는것이다. 그렇게하면, 비디오 영상 추적은 점점 부드럽게 된다.


```python
def img_pipeline(img):
    
    global window_search
    global left_fit_prev
    global right_fit_prev
    global frame_count
    global curve_radius
    global offset
        
    # load camera matrix and distortion matrix
    camera = pickle.load(open( "camera_matrix.pkl", "rb" ))
    mtx = camera['mtx']
    dist = camera['dist']
    camera_img_size = camera['imagesize']
    
    #correct lens distortion
    undist = distort_correct(img,mtx,dist,camera_img_size)
    # get binary image
    binary_img = binary_pipeline(undist)
    #perspective transform
    birdseye, inverse_perspective_transform = warp_image(binary_img)
    
    if window_search:
        #window_search = False
        #window search
        left_fit,right_fit = track_lanes_initialize(birdseye)
        #store values
        left_fit_prev = left_fit
        right_fit_prev = right_fit
        
    else:
        #load values
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        #search in margin of polynomials
        left_fit,right_fit,leftx,lefty,rightx,righty = track_lanes_update(birdseye, left_fit,right_fit)
    
    #save values
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    
    #draw polygon
    processed_frame = lane_fill_poly(birdseye, undist, left_fit, right_fit)
    
    #update ~twice per second
    if frame_count==0 or frame_count%15==0:
        #measure radii
        curve_radius = measure_curve(birdseye,left_fit,right_fit)
        #measure offset
        offset = vehicle_offset(undist, left_fit, right_fit)
    
        
    #printing information to frame
    font = cv.FONT_HERSHEY_TRIPLEX
    processed_frame = cv.putText(processed_frame, 'Radius: '+str(curve_radius)+' m', (30, 40), font, 1, (0,255,0), 2)
    processed_frame = cv.putText(processed_frame, 'Offset: '+str(offset)+' m', (30, 80), font, 1, (0,255,0), 2)
   
    frame_count += 1
    return processed_frame
```

# 실행
### case 1 : 사진


```python
filenames = os.listdir("test_images/")
global window_search
global frame_count
for filename in filenames:
    frame_count = 15
    window_search = True
    image = mpimg.imread('test_images/'+filename)
    lane_image = img_pipeline(image)
    mpimg.imsave('output_images/lane_'+filename,lane_image)

Image('output_images/lane_test4.jpg')
```

    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)
    (1280, 720)





![jpeg](https://i.imgur.com/hNgMEsc.jpg)



### case 2 : 비디오


```python
from moviepy.video.fx.all import crop

global window_search 
global frame_count
window_search = True
frame_count = 0

#chicago footage
for filename in ['processed_extra.mp4']:
    clip = VideoFileClip('videos/'+filename)#.subclip((3,25),(3,35))
    #clip_crop = crop(clip, x1=320, y1=0, x2=1600, y2=720)
    out= clip.fl_image(img_pipeline)
    #out = clip_crop.fl_image(img_pipeline)
    out.write_videofile('videos/processed_'+filename, audio=False, verbose=False)
    print('Success!')
```

    (1280, 720)
    (1280, 720)


    t:   0%|          | 0/251 [00:00<?, ?it/s, now=None]

    Moviepy - Building video videos/processed_processed_extra.mp4.
    Moviepy - Writing video videos/processed_processed_extra.mp4
    
    (1280, 720)
    (1280, 720)


    t:   1%|          | 2/251 [00:05<10:40,  2.57s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   1%|          | 3/251 [00:10<13:28,  3.26s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   2%|▏         | 4/251 [00:15<15:44,  3.82s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   2%|▏         | 5/251 [00:20<17:16,  4.21s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   2%|▏         | 6/251 [00:26<19:14,  4.71s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   3%|▎         | 7/251 [00:31<19:48,  4.87s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   3%|▎         | 8/251 [00:36<19:40,  4.86s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   4%|▎         | 9/251 [00:38<16:09,  4.01s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   4%|▍         | 10/251 [00:42<16:55,  4.21s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   4%|▍         | 11/251 [00:47<17:49,  4.45s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   5%|▍         | 12/251 [00:52<17:56,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   5%|▌         | 13/251 [00:57<18:31,  4.67s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   6%|▌         | 14/251 [01:02<18:07,  4.59s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   6%|▌         | 15/251 [01:06<18:06,  4.60s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   6%|▋         | 16/251 [01:08<14:25,  3.68s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   7%|▋         | 17/251 [01:11<14:10,  3.63s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   7%|▋         | 18/251 [01:16<15:25,  3.97s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   8%|▊         | 19/251 [01:21<16:37,  4.30s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   8%|▊         | 20/251 [01:26<17:35,  4.57s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   8%|▊         | 21/251 [01:31<17:55,  4.68s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   9%|▉         | 22/251 [01:35<17:10,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:   9%|▉         | 23/251 [01:39<16:24,  4.32s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  10%|▉         | 24/251 [01:43<16:20,  4.32s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  10%|▉         | 25/251 [01:48<16:29,  4.38s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  10%|█         | 26/251 [01:53<16:48,  4.48s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  11%|█         | 27/251 [01:57<16:47,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  11%|█         | 28/251 [02:02<17:28,  4.70s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  12%|█▏        | 29/251 [02:06<15:41,  4.24s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  12%|█▏        | 30/251 [02:11<16:32,  4.49s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  12%|█▏        | 31/251 [02:17<18:21,  5.01s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  13%|█▎        | 32/251 [02:21<17:50,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  13%|█▎        | 33/251 [02:27<18:42,  5.15s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  14%|█▎        | 34/251 [02:33<18:55,  5.23s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  14%|█▍        | 35/251 [02:38<19:15,  5.35s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  14%|█▍        | 36/251 [02:43<18:16,  5.10s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  15%|█▍        | 37/251 [02:48<18:50,  5.28s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  15%|█▌        | 38/251 [02:53<18:01,  5.08s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  16%|█▌        | 39/251 [02:58<17:27,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  16%|█▌        | 40/251 [03:03<17:22,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  16%|█▋        | 41/251 [03:08<17:30,  5.00s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  17%|█▋        | 42/251 [03:13<17:31,  5.03s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  17%|█▋        | 43/251 [03:18<17:08,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  18%|█▊        | 44/251 [03:21<15:11,  4.40s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  18%|█▊        | 45/251 [03:25<15:03,  4.38s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  18%|█▊        | 46/251 [03:30<15:16,  4.47s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  19%|█▊        | 47/251 [03:35<15:32,  4.57s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  19%|█▉        | 48/251 [03:39<14:49,  4.38s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  20%|█▉        | 49/251 [03:43<15:09,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  20%|█▉        | 50/251 [03:48<15:26,  4.61s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  20%|██        | 51/251 [03:53<15:41,  4.71s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  21%|██        | 52/251 [03:58<15:32,  4.69s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  21%|██        | 53/251 [04:03<16:16,  4.93s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  22%|██▏       | 54/251 [04:07<15:13,  4.64s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  22%|██▏       | 55/251 [04:12<15:34,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  22%|██▏       | 56/251 [04:17<15:41,  4.83s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  23%|██▎       | 57/251 [04:18<11:57,  3.70s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  23%|██▎       | 58/251 [04:23<13:09,  4.09s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  24%|██▎       | 59/251 [04:28<13:31,  4.23s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  24%|██▍       | 60/251 [04:33<13:51,  4.35s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  24%|██▍       | 61/251 [04:37<14:19,  4.52s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  25%|██▍       | 62/251 [04:43<15:06,  4.80s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  25%|██▌       | 63/251 [04:49<16:03,  5.12s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  25%|██▌       | 64/251 [04:54<15:53,  5.10s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  26%|██▌       | 65/251 [04:59<15:58,  5.15s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  26%|██▋       | 66/251 [05:04<15:54,  5.16s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  27%|██▋       | 67/251 [05:09<15:38,  5.10s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  27%|██▋       | 68/251 [05:14<15:39,  5.13s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  27%|██▋       | 69/251 [05:19<15:11,  5.01s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  28%|██▊       | 70/251 [05:24<14:39,  4.86s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  28%|██▊       | 71/251 [05:29<14:48,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  29%|██▊       | 72/251 [05:33<14:23,  4.83s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  29%|██▉       | 73/251 [05:38<13:54,  4.69s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  29%|██▉       | 74/251 [05:44<15:04,  5.11s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  30%|██▉       | 75/251 [05:48<14:30,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  30%|███       | 76/251 [05:54<14:42,  5.04s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  31%|███       | 77/251 [05:59<14:55,  5.15s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  31%|███       | 78/251 [06:04<14:23,  4.99s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  31%|███▏      | 79/251 [06:08<13:53,  4.85s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  32%|███▏      | 80/251 [06:13<13:52,  4.87s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  32%|███▏      | 81/251 [06:17<12:50,  4.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  33%|███▎      | 82/251 [06:21<12:43,  4.52s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  33%|███▎      | 83/251 [06:26<12:56,  4.62s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  33%|███▎      | 84/251 [06:31<13:06,  4.71s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  34%|███▍      | 85/251 [06:36<13:31,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  34%|███▍      | 86/251 [06:41<13:26,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  35%|███▍      | 87/251 [06:46<13:04,  4.79s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  35%|███▌      | 88/251 [06:51<13:25,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  35%|███▌      | 89/251 [06:57<13:45,  5.09s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  36%|███▌      | 90/251 [07:02<13:59,  5.22s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  36%|███▋      | 91/251 [07:08<14:04,  5.28s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  37%|███▋      | 92/251 [07:13<14:29,  5.47s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  37%|███▋      | 93/251 [07:18<13:26,  5.10s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  37%|███▋      | 94/251 [07:23<13:42,  5.24s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  38%|███▊      | 95/251 [07:28<13:26,  5.17s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  38%|███▊      | 96/251 [07:32<12:26,  4.82s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  39%|███▊      | 97/251 [07:37<12:07,  4.73s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  39%|███▉      | 98/251 [07:42<12:16,  4.81s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  39%|███▉      | 99/251 [07:46<12:04,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  40%|███▉      | 100/251 [07:51<12:05,  4.80s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  40%|████      | 101/251 [07:56<11:53,  4.75s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  41%|████      | 102/251 [08:01<12:09,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  41%|████      | 103/251 [08:06<12:07,  4.92s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  41%|████▏     | 104/251 [08:12<12:34,  5.13s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  42%|████▏     | 105/251 [08:17<12:17,  5.05s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  42%|████▏     | 106/251 [08:21<11:55,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  43%|████▎     | 107/251 [08:26<11:27,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  43%|████▎     | 108/251 [08:30<10:52,  4.56s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  43%|████▎     | 109/251 [08:34<10:40,  4.51s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  44%|████▍     | 110/251 [08:39<11:00,  4.68s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  44%|████▍     | 111/251 [08:45<11:34,  4.96s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  45%|████▍     | 112/251 [08:49<11:08,  4.81s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  45%|████▌     | 113/251 [08:54<11:16,  4.91s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  45%|████▌     | 114/251 [08:59<10:55,  4.78s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  46%|████▌     | 115/251 [09:04<10:50,  4.78s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  46%|████▌     | 116/251 [09:08<10:44,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  47%|████▋     | 117/251 [09:13<10:47,  4.83s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  47%|████▋     | 118/251 [09:18<10:36,  4.79s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  47%|████▋     | 119/251 [09:22<09:45,  4.43s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  48%|████▊     | 120/251 [09:27<10:01,  4.59s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  48%|████▊     | 121/251 [09:32<10:19,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  49%|████▊     | 122/251 [09:36<09:47,  4.56s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  49%|████▉     | 123/251 [09:42<10:25,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  49%|████▉     | 124/251 [09:47<10:24,  4.92s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  50%|████▉     | 125/251 [09:51<09:47,  4.66s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  50%|█████     | 126/251 [09:52<07:34,  3.64s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  51%|█████     | 127/251 [09:56<08:01,  3.88s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  51%|█████     | 128/251 [10:01<08:11,  3.99s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  51%|█████▏    | 129/251 [10:05<08:40,  4.27s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  52%|█████▏    | 130/251 [10:11<09:08,  4.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  52%|█████▏    | 131/251 [10:11<06:40,  3.33s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  53%|█████▎    | 132/251 [10:16<07:19,  3.70s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  53%|█████▎    | 133/251 [10:21<08:04,  4.10s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  53%|█████▎    | 134/251 [10:26<08:22,  4.30s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  54%|█████▍    | 135/251 [10:31<08:45,  4.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  54%|█████▍    | 136/251 [10:35<08:43,  4.55s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  55%|█████▍    | 137/251 [10:40<08:56,  4.71s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  55%|█████▍    | 138/251 [10:45<08:51,  4.70s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  55%|█████▌    | 139/251 [10:50<08:52,  4.76s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  56%|█████▌    | 140/251 [10:55<08:50,  4.78s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  56%|█████▌    | 141/251 [10:59<08:27,  4.62s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  57%|█████▋    | 142/251 [11:03<08:09,  4.49s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  57%|█████▋    | 143/251 [11:08<08:27,  4.70s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  57%|█████▋    | 144/251 [11:14<09:10,  5.14s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  58%|█████▊    | 145/251 [11:20<09:08,  5.17s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  58%|█████▊    | 146/251 [11:24<08:30,  4.87s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  59%|█████▊    | 147/251 [11:29<08:24,  4.86s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  59%|█████▉    | 148/251 [11:35<08:59,  5.24s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  59%|█████▉    | 149/251 [11:40<08:50,  5.20s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  60%|█████▉    | 150/251 [11:45<08:54,  5.29s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  60%|██████    | 151/251 [11:50<08:19,  4.99s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  61%|██████    | 152/251 [11:54<07:59,  4.85s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  61%|██████    | 153/251 [11:59<07:53,  4.83s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  61%|██████▏   | 154/251 [12:04<07:58,  4.93s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  62%|██████▏   | 155/251 [12:09<07:47,  4.87s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  62%|██████▏   | 156/251 [12:13<07:21,  4.64s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  63%|██████▎   | 157/251 [12:16<06:42,  4.28s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  63%|██████▎   | 158/251 [12:21<06:51,  4.43s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  63%|██████▎   | 159/251 [12:27<07:10,  4.68s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  64%|██████▎   | 160/251 [12:29<06:11,  4.08s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  64%|██████▍   | 161/251 [12:35<06:53,  4.60s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  65%|██████▍   | 162/251 [12:39<06:40,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  65%|██████▍   | 163/251 [12:40<05:09,  3.51s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  65%|██████▌   | 164/251 [12:46<05:46,  3.98s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  66%|██████▌   | 165/251 [12:50<05:44,  4.00s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  66%|██████▌   | 166/251 [12:53<05:35,  3.95s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  67%|██████▋   | 167/251 [12:58<05:51,  4.19s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  67%|██████▋   | 168/251 [13:03<06:05,  4.40s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  67%|██████▋   | 169/251 [13:08<06:08,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  68%|██████▊   | 170/251 [13:12<06:04,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  68%|██████▊   | 171/251 [13:17<06:02,  4.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  69%|██████▊   | 172/251 [13:22<06:15,  4.76s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  69%|██████▉   | 173/251 [13:27<06:16,  4.83s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  69%|██████▉   | 174/251 [13:31<05:55,  4.62s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  70%|██████▉   | 175/251 [13:36<06:01,  4.76s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  70%|███████   | 176/251 [13:42<06:07,  4.90s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  71%|███████   | 177/251 [13:45<05:37,  4.56s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  71%|███████   | 178/251 [13:49<05:21,  4.41s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  71%|███████▏  | 179/251 [13:54<05:16,  4.40s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  72%|███████▏  | 180/251 [13:58<05:11,  4.39s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  72%|███████▏  | 181/251 [14:03<05:21,  4.60s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  73%|███████▎  | 182/251 [14:04<04:03,  3.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  73%|███████▎  | 183/251 [14:08<04:01,  3.55s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  73%|███████▎  | 184/251 [14:13<04:25,  3.96s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  74%|███████▎  | 185/251 [14:18<04:36,  4.20s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  74%|███████▍  | 186/251 [14:23<04:51,  4.49s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  75%|███████▍  | 187/251 [14:28<05:05,  4.78s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  75%|███████▍  | 188/251 [14:33<04:53,  4.66s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  75%|███████▌  | 189/251 [14:37<04:42,  4.56s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  76%|███████▌  | 190/251 [14:41<04:27,  4.39s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  76%|███████▌  | 191/251 [14:45<04:21,  4.36s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  76%|███████▋  | 192/251 [14:50<04:22,  4.45s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  77%|███████▋  | 193/251 [14:54<04:16,  4.42s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  77%|███████▋  | 194/251 [14:59<04:25,  4.65s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  78%|███████▊  | 195/251 [15:05<04:34,  4.91s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  78%|███████▊  | 196/251 [15:10<04:27,  4.86s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  78%|███████▊  | 197/251 [15:14<04:07,  4.59s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  79%|███████▉  | 198/251 [15:18<04:00,  4.54s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  79%|███████▉  | 199/251 [15:23<04:07,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  80%|███████▉  | 200/251 [15:27<03:46,  4.44s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  80%|████████  | 201/251 [15:32<03:50,  4.60s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  80%|████████  | 202/251 [15:38<04:08,  5.07s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  81%|████████  | 203/251 [15:43<03:56,  4.92s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  81%|████████▏ | 204/251 [15:47<03:39,  4.67s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  82%|████████▏ | 205/251 [15:51<03:29,  4.54s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  82%|████████▏ | 206/251 [15:57<03:37,  4.84s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  82%|████████▏ | 207/251 [16:02<03:35,  4.90s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  83%|████████▎ | 208/251 [16:06<03:21,  4.68s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  83%|████████▎ | 209/251 [16:11<03:25,  4.90s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  84%|████████▎ | 210/251 [16:16<03:20,  4.89s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  84%|████████▍ | 211/251 [16:19<02:52,  4.30s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  84%|████████▍ | 212/251 [16:22<02:33,  3.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  85%|████████▍ | 213/251 [16:26<02:31,  4.00s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  85%|████████▌ | 214/251 [16:31<02:34,  4.18s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  86%|████████▌ | 215/251 [16:36<02:40,  4.46s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  86%|████████▌ | 216/251 [16:41<02:38,  4.53s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  86%|████████▋ | 217/251 [16:45<02:31,  4.45s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  87%|████████▋ | 218/251 [16:50<02:32,  4.63s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  87%|████████▋ | 219/251 [16:55<02:31,  4.73s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  88%|████████▊ | 220/251 [17:00<02:30,  4.84s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  88%|████████▊ | 221/251 [17:04<02:21,  4.71s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  88%|████████▊ | 222/251 [17:08<02:09,  4.46s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  89%|████████▉ | 223/251 [17:12<02:01,  4.35s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  89%|████████▉ | 224/251 [17:17<02:00,  4.45s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  90%|████████▉ | 225/251 [17:22<01:58,  4.55s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  90%|█████████ | 226/251 [17:27<02:01,  4.84s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  90%|█████████ | 227/251 [17:33<01:58,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  91%|█████████ | 228/251 [17:37<01:49,  4.77s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  91%|█████████ | 229/251 [17:41<01:43,  4.69s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  92%|█████████▏| 230/251 [17:47<01:43,  4.94s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  92%|█████████▏| 231/251 [17:50<01:30,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  92%|█████████▏| 232/251 [17:55<01:25,  4.49s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  93%|█████████▎| 233/251 [17:59<01:21,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  93%|█████████▎| 234/251 [18:05<01:20,  4.76s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  94%|█████████▎| 235/251 [18:08<01:10,  4.38s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  94%|█████████▍| 236/251 [18:13<01:07,  4.52s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  94%|█████████▍| 237/251 [18:17<01:00,  4.30s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  95%|█████████▍| 238/251 [18:21<00:54,  4.20s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  95%|█████████▌| 239/251 [18:26<00:52,  4.34s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  96%|█████████▌| 240/251 [18:31<00:51,  4.66s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  96%|█████████▌| 241/251 [18:35<00:45,  4.55s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  96%|█████████▋| 242/251 [18:38<00:34,  3.88s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  97%|█████████▋| 243/251 [18:43<00:35,  4.47s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  97%|█████████▋| 244/251 [18:49<00:33,  4.73s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  98%|█████████▊| 245/251 [18:54<00:29,  4.96s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  98%|█████████▊| 246/251 [18:58<00:23,  4.72s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  98%|█████████▊| 247/251 [19:03<00:19,  4.75s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  99%|█████████▉| 248/251 [19:08<00:14,  4.74s/it, now=None]

    (1280, 720)
    (1280, 720)


    t:  99%|█████████▉| 249/251 [19:12<00:08,  4.48s/it, now=None]

    (1280, 720)
    (1280, 720)


    t: 100%|█████████▉| 250/251 [19:16<00:04,  4.50s/it, now=None]

    (1280, 720)
    (1280, 720)


    t: 100%|██████████| 251/251 [19:19<00:00,  4.02s/it, now=None]

    (1280, 720)
    (1280, 720)


                                                                  

    Moviepy - Done !
    Moviepy - video ready videos/processed_processed_extra.mp4
    Success!



```python
video = 'videos/processed_processed_extra.mp4'
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video))
```





<video width="960" height="540" controls>
  <source src="videos/processed_processed_extra.mp4">
</video>




---
