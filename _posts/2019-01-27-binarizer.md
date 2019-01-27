
> "타겟 값이 순서형인데 분류문제로 풀어도 될까?"
![](https://t1.daumcdn.net/cfile/tistory/272245495820906B04)  

예를 들어 타겟 값이 암의 병기를 나타내는 컬럼일 때,  
데이터는 1, 2, 3 순서형이다.  

https://www.kaggle.com/c/petfinder-adoption-prediction/data  
이 대회의 타겟 값 AdoptionSpeed는 순서형으로  
데이터는 1, 2, 3 의 형식을 따르고 있다.  

이렇게 순서형/서열형 타겟값을 따를 경우에는 타겟값을 잘 반영할 수 있는 모델을 선택해야 한다.  

회귀처럼 풀었을 경우 예측값이 1.5일 경우 1에 가까울 지 2에 가까울 지 판단하기 어렵다.  
분류로 풀었을 경우 1과 2의 상대적 크고 작음을 고려할 수 없다.  
-> 서열척도 성질이 균등한 동일간격을 가지고 있지 않기 때문이다.  

---

순서형 로짓 회귀 모델을 알아보자.  

[해당 논문](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf)에 따르면 데이터셋이 요런 형식 이어야함  

![](https://i.imgur.com/49O7hy1.png) 

```python
from sklearn import preprocessing
import numpy as np
import pandas as pd
```
예제 파일은 위의 캐글에서 받을 수 있다.  
위의 데이터셋을 나누기 위해서 서열을 리스트로 만들어  
하나씩 늘려가면서 만들면 될 것 같다.  

```python
lst = [1, 2, 3]
datasets = {}
for idx range(1, len(lst):
  classes = lst[:idx]
```
위의 클래스에 따라서 데이터셋을 만들어 주려고 한다.

```python
df.iloc[:,-1].isin(classes)
```
이제 이 친구들을 바이너리화 해줘야 한다.
```python
lb = preprocessing.LabelBinarizer()
lb.fit_transform(df)
```
변환 된 것을 크기에 맞게 변환시켜 준다  
```python
y_train = np.concatenate(df)
```
이렇게 만들어진 것들을 차례씩 datasets에 넣어주자
```python
datasets[idx] = {'x_train' : x_train, 'y_train':y_train}
```
이제 위 논문의 datasets을 만들었다.  
모델에 넣고 돌려보자  
```python
models = {}
for k, v in datasets.items():
  clf = RandomForestClassifier()
  models[k] = clf.fit(v['x_train'], v['y_train'])
```
이렇게 하면 모델별로 수치가 나온다.  
