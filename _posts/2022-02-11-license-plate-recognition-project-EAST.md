---
layout: post
title: 차량 번호판 인식 실전 프로젝트 - License Plate Detection with EAST
subtitle: Custom Dataset 실전 프로젝트 실습 2 - EAST를 이용한 License Plate Detection 모델(Custom Dataset) 학습 (진행중)
cover-img: /assets/img/car_recognition.jpg
thumbnail-img: /assets/img/posting2/east_detection.jpg
share-img: /assets/img/car_recognition.jpg
tags: [text detection, custom dataset, East Detection]
---

# License Plate Detection - 차량 번호판 인식 실전 프로젝트

이 프로젝트는 Inflearn [차량 번호판 인식 프로젝트와 TensorFlow로 배우는 딥러닝 영상인식 올인원](https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard) 수업의 결과물을 정리한 내용입니다. [1]

CRAFT 모델에 이어, **EAST 모델을 활용하여 차량 번호판에서 텍스트 영역을 검출**하는 프로젝트를 진행해보았습니다. (기존 EAST 코드를 강사님께서 Tensorflow2 version으로 개선하여 작성하신 소스 코드를 활용했습니다. [Github 링크](https://github.com/solaris33/EAST-tf2))

## EAST (Efficient and Accuracy Scene Text)

중국 기업 MEGVII에서 2017년에 제안한 논문입니다. EAST 모델은 기존 기법들보다 간결한 구조를 취하고 있다는 장점을 가지고 있습니다. EAST에서 사용하는 모델은 Stem Layer(Feature extraction layer), Feature-merging branch layer, Output layer 이렇게 세 가지 형태로 구성됩니다. 또한 EAST 모델은 기존 모델 대비 F1 score도 높고, 속도도 빨라서 획기적인 text model로 알려져 있습니다. [2]
 
![Untitled 1](../assets/img/posting2/Untitled%201.jpg) 

그림 1 - EAST 모델의 아키텍쳐 [3]

![Untitled 2](../assets/img/posting2/Untitled%202.png)  

그림 2 - EAST 모델의 성능지표 [4]


## EAST 모델을 License Plate Detection 데이터에 맞게 Fine-Tuning하기

 
### 실습 환경
* Google Colab (GPU - Tesla P100-PCIE)

![Untitled 3](../assets/img/posting2/Untitled%203.png)

그림 3 - Google colab 환경

## 진행 방법

## 1. 범용적인 Dataset에 대해서 학습하기 
1 ) Colab에서 TF2.0으로 수정된 EAST 코드를 다운 받습니다.  
* ICDAR 2015 Dataset을 다운받아서 트레이닝을 진행하고, test evaluation을 할 수 있도록 되어 있습니다.  

2 ) Github 문서를 참고하면 Directory 구조는 아래 사진과 같이 될 것을 필요로 합니다. [5]

![Untitled 4](../assets/img/posting2/Untitled%204.png)  

그림 4 - East 모델의 디렉토리 구조

3 ) Training을 위한 ICDAR 2015 Dataset을 공식 홈페이지([링크](https://rrc.cvc.uab.es/?ch=4&com=downloads))에서 다운로드 받은 후, 해당 Dataset을 Colab에 업로드합니다. Dataset은 아래와 같이 구성되어 있습니다.   
  
* train_data
    * 1000장의 ICDAR 이미지에 대한 jpg 파일과 정답 txt 파일이 들어가있습니다.
    * 정답 txt 파일에는 8개의 꼭짓점 벡터와 그 안에 정답 label이 있습니다.
* test_data
    * 500장의 test 이미지만 들어있습니다.

4 ) Training 하기
* 다운 받은 Dataset의 압축을 푼 후에 github 문서의 가이드를 따라서, 아래와 같은 명령어를 입력하여 training을 수행합니다.

```python
!python train.py 
--training_data_paht="./data/ICDAR2015/train_data/" \
--checkpoint_path="/content/drive/MyDrive/Colab Notebooks/인프런_컴퓨터비전_올인원/6강/east_resnet_50_rbox"
```
이 때  Training의 시간이 매우 길기 때문에 colab 연결이 끊어질 경우를 대비해서, google drive를 mount하여 checkpoint 파일이 드라이브에 저장되도록 했습니다.
따라서 런타임이 끊어지더라도 ckpt 파일이 구글 드라이브에 저장되어, 해당 ckpt 이후부터 training을 이어서 할 수 있습니다.  

![Untitled 5](../assets/img/posting2/Untitled%205.png)  
그림 5 - google drive에 mount

다음과 같이 런타임이 끊어졌지만, drive에 ckpt 파일이 저장되어 해당 부분을 불러와 그 이 후부터 다시 training을 진행할 수 있었습니다. 

![Untitled 6](../assets/img/posting2/Untitled%206.png)  
그림 6 - checkpoint 파일의 복구
#  

     
## 2. EAST 모델을 License Plate Detection 데이터에 맞게 Fine-Tuning 하기

1 ) Colab에서 TF2.0으로 수정된 EAST 코드를 다운 받습니다. 

2 ) license_plate_detection_data.zip 파일을 업로드 한 후 압축을 풀어줍니다. image와 annotaion 파일들이 들어갑니다. 

* 80%는 training directory, 20%를 test data로 split을 진행합니다. 각각을 train_data, test_data 폴더에 넣어줍니다. 

3 ) license_plate_detection_data에 맞춰서 새롭게 training을 진행합니다. (Fine-Tuning)

이 때 data_preprocessor.py에서 수정이 필요합니다. 해당 코드는 ICDAR data에 맞춰 작성되었기 때문에, gt_로 replace 하는 부분을 삭제하고, load_annotaion( ) 함수에서 데이터를 띄어쓰기 기준으로 split 하는 함수를 작성해주어야 합니다. 

4 ) train을 진행합니다. training_data_path는 80%를 모아놓은 경로를 설정하고, checkpoint_path도 학습 결과를 저장할 수 있는 경로를 설정합니다. 

5 ) 10만번의 ckpt 파일이 생기면, test_data_output 폴더를 만들어줍니다.  
eval.py 실행할 때, test할 이미지를 test_data_path로 잡아주고 model_path는 학습이 끝난 checkpoint를 가지고 있는 곳으로 경로를 잡아줍니다. 이 checkpoint 파일로 prediction한 결과가 저장되는 것을 test_data_ouput 폴더에 지정해줍니다. 

6 ) prediction이 진행됩니다. test_data_output 폴더를 보면 이미지에 대한 License plate 모델이prediction 한 것과 위치 포지션 txt 파일이 같이 생성됩니다. 

7 ) prediction이 끝난 후 test_data_ouput 폴더를 압축하여 다운로드 받아줍니다. local에서 결과를 확인해봅니다. 45장의 test 이미지 중 1장을 제외하고 정확하게 예측을 합니다.




# References

[1] [https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard](https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard)

[2] [https://github.com/argman/EAST](https://github.com/argman/EAST)

[3] [https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)

[4] [https://www.researchgate.net/publication/316015737_EAST_An_Efficient_and_Accurate_Scene_Text_Detector](https://www.researchgate.net/publication/316015737_EAST_An_Efficient_and_Accurate_Scene_Text_Detector)

[5] [https://github.com/solaris33/EAST-tf2](https://github.com/solaris33/EAST-tf2)





