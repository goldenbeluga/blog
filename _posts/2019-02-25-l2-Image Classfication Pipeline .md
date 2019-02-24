L2-Image Classfication Pipeline  

# Distance  
![]({{site.url}}/assets/img/Image.png )
![]({{site.url}}/assets/img/Image [1].png )  
L1으로 계산하면 두 image의 픽셀값 차이의 절대값을 전부 더하면 된다.  
![]({{site.url}}/assets/img/Image [2].png )  
> L1 마름모꼴은 어떻게 그려질까?  
굵은 선은 총 거리를 뜻함  
![]({{site.url}}/assets/img/Image [3].png )   
최단 거리를 2라고 하면 (2,0), (1,1) 등의 결과로 나타난다.  
예를 들어 (1,0),(0,1)의 L1거리는 (1,1)이 된다.  
|1-0| + |0-1| = 2 이다.   

> 언제 L1을 써야하고 L2를 써야할까?  
-> problem dependent  
좌표시스템에 달라진다.  
벡터가 근속연수, 봉급처럼 각 요소가 의미를 가지면 L1을 써라.  
"왜인지는 설명 안하심ㅠㅠ"  
![]({{site.url}}/assets/img/Image [4].png )  


거리만 근사하면 분류해버리기 때문에 잘못된 판단을 할 수 있음  
Original 과 각각의 사진과의 거리가 전부 같음  
(교수가 일부러 거리가 같아지게 이미지를 만듦)  
따라서 L1/L2 이미지에 쓰이기 적절하지 않다.  

이미지 분류에서는 KNN이 쓰지 않음  
	1. 각각의 sample과의 거리 계산을 해야해서 너무 느림  
	2. 차원의 저주(curse of dimensionality)  
![]({{site.url}}/assets/img/Image [5].png)   

  
knn이 잘 작동하기 위해선 전체 공간을 조밀하게 커버할 만큼의 충분한 sample이 필요함  
만약 3차원일 때 중간에 뻥 뚫려있을 경우, test의 한 점이 뚫려 있는 곳에 들어왔을 경우를 생각해보자  
그러면 뚫려있지 않았을 경우는 가장 가까운 점의 거리가 1이였는데 뚫려있을 때의 경우는 가까운 점과의 거리가 200이 되었다.  
그렇다면 200이 되는 점의 label을 따라야 할까? 확실히 그 label임이라 말하기 힘들 것이다.  
여기에 관한 내용은 핸즈온 머신러닝 p.271을 읽으면 더 쉽게 이해된다.  


# Setting Hyperparameters  
![]({{site.url}}/assets/img/Image [6].png)  
![]({{site.url}}/assets/img/Image [7].png )  

딥러닝에선 학습자체가 계산량이 너무 커서 Cross-Validation안 씀  
작은 데이터일 때 사용한다.  

# Linear Classifiers
![]({{site.url}}/assets/img/Image [8].png )  
![]({{site.url}}/assets/img/Image [9].png )  

X는 Image, W는 weight, b는 bias  
![]({{site.url}}/assets/img/Image [10].png )  
흰 색이 X를 뜻함, W는 임의로 나온 숫자  

## Linear Classifiers 의 한계점  
![]({{site.url}}/assets/img/Image [11].png )  
W로 그림을 저렇게 나왔음  
그런데 문제가 있음  
Linear classifier는 분류하는데 하나의 템플릿(W)으로 판단  
horse를 판단한 애의 W로 그림을 그렸더니 얼굴이 양쪽에 있다.  
한 클래스내에 다양한 특성이 존재하지만 각 카테고리를 인식하기 위한 템플릿은 하나 밖에 없음  

> Neural Network같은 복잡한 모델은 더 잘 나올 것이다.  
![]({{site.url}}/assets/img/Image [12].png )  

(1,1)은 0보다 큰 픽셀이 2개 이니깐 even -> Red  
(-1,1)은 0보다 큰 픽셀이 1개 이니깐 odd -> Blue  
이런 홀수/짝수 같은 반전성/패리티 문제(parity problem)에선Linear classifier는 한 직선으로 구분못 함  

