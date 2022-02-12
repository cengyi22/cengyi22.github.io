---
layout: post
title: 차량 번호판 인식 실전 프로젝트 - License Plate Detection
subtitle: Custom Dataset 실전 프로젝트 실습 1 - CRAFT를 이용한 License Plate Detection 모델(Custom Dataset) 학습 (진행중)
cover-img: /assets/img/car_recognition.jpg
thumbnail-img: /assets/img/keras_ocr.jpg
share-img: /assets/img/car_recognition.jpg
tags: [text detection, custom dataset, craft]
---

# License Plate Detection - 차량 번호판 인식 실전 프로젝트

이 프로젝트는 Inflearn [차량 번호판 인식 프로젝트와 TensorFlow로 배우는 딥러닝 영상인식 올인원](https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard) 수업의 결과물을 정리한 내용입니다. 

## Text Detection 문제영역 소개
이미지 내에 텍스트가 존재하는 영역의 위치정보를 Bounding Box로 찾는 문제 영역입니다. 의미 정보와 상관 없이 Text가 어느 위치에 있는 지를 찾아줍니다.   
일반적인 Object Detection과의 차이점은 Rotation Angle까지 고려해서 Detection을 하기 때문에, Rotation Angle을 고려하지 않는 Object Detection보다 좀 더 정확하게 진행할 수 있습니다.

![Untitled](../assets/img/Untitled.png)

그림 1 - Text Detection 모델의 결과 - (Text의 위치 
정보에 대한 Bounding Box) [1]

Text Detection 기술은 교통 카메라에서 자동차 번호판 인식, 시각적 검색 등 다양한 분야의 현실세계 문제에 활발히 사용 되고 있습니다.

![Untitled 1](../assets/img/Untitled%201.png)  


그림 2 - Text Detection 기술을 활용한 자동차 번호판 인식 [2]

![Untitled 2](../assets/img/Untitled%202.png)

그림 3 - News Video에서의 Text Detection [3]

## OCR Open Dataset1 - ICDAR 2015

ICDAR(Incidental Scene Text Dataset)은 1500장 이미지로 구성되어 있으며, 1000장은 Training 나머지는 testing으로 사용합니다. 4개의 꼭지점 좌표가 annotation으로 되어 있습니다. 2013년 Dataset은 229장입니다. [4]

![Untitled 3](../assets/img/Untitled%203.png)

그림 4 - Google glass로 촬영한 icdar 샘플 사진


# OCR Open Dataset2 - COCO-Text Dataset

COCO-Text Dataset은 63686장의 이미지와 annotation들이 있습니다. annotation은 AABB 형태로 되어 있는데 RBOX의 special case로 rotaion angle이 0인 정방향의 label 형태로 볼 수 있습니다. 

![Untitled 4](../assets/img/Untitled%204.png)

그림 5 - COCO Dataset 샘플 사진 [5]

# Car License Plate recognition project 

차량 번호판 인식에 대한 연구는 이전의 hand crafted를 이용한 컴퓨터 비전 분야에서 활발하게 진행되고 있는 주제입니다. 그러나 최근에는 고속 영상에서의 번호 인식 등을 위해 딥러닝 기술과 접목되어 연구되고 있습니다. 

성능이 높은 Text Detection 모델인 **CRAFT 모델을 활용하여 차량 번호판에서 텍스트 영역을 검출**하는 프로젝트를 진행해보았습니다.   
(**CRAFT 공식 문서와 [6] Github를 참고 하였고 [7], Inflearn AI-School 강사님의 커리큘럼을 따라**해보았습니다. [8])

### License Plate Dataset에 대해 CRAFT Detector 학습
* 기존 CRAFT Dataset을 Custom Dataset인 License Plate Detection Dataset에 적합한 파라미터로 Fine-Tuning을 해보는 프로젝트였습니다. 
[keras-ocr의 예제코드](https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_detector.html)를 참조하되 get_license_plate_detector_dataset이라는 새로운 function을 만들고, function의 반환값을 get_icdar_2013_detector_dataset과 같은 shape으로 만든 후 나머지 과정을 똑같이 진행합니다.

### 강의에서 제공된 License Plate Dataset 
* image 
    * 222장의 차량번호판이 포함된 이미지 
* Ground truth Annotation 
    * License Plate의 꼭짓점 절대좌표가 띄어쓰기로 split 되어서 시계방향으로 x_1 y_1 x_2 y_2 x_3 y_3 x_4 y_4 label 의 형태

### 실습 환경
* Google Colab이용 (Tesla P100-PCIE)

![Untitled 5](../assets/img/Untitled%205.png)

그림 6 - Google colab 환경
### 진행 방법
* 1. 제공된 Dataset에서 jpg 파일과 txt 파일에서 필요한 정보들을 parsing해서 keras-ocr 예제의 get_license_plate_detector_dataset 함수의 반환값과 같은 형태로 만들어줍니다.
* 2. license plate detector Dataset에 맞게 CRAFT 파라미터를 FINE-Tuning 합니다.

### 정리
* Custom dataset의 일종인 license_plate_detection 데이터를 피딩할 수 있는 custom 함수를 만들어서, plate detection이 잘 검출될 수 있는 방향성으로 다시 fine tuning을 시켜주었습니다. 재학습이 끝난 모델에 대해 prediction을 해서 정성적으로 만든 모델이 동작하는지를 분석했습니다.  

# References

[1] [https://docs.aws.amazon.com/ko_kr/rekognition/latest/dg/text-detection.html](https://docs.aws.amazon.com/ko_kr/rekognition/latest/dg/text-detection.html)

[2] [https://customers.pyimagesearch.com/lesson-sample-segmenting-characters-from-license-plates/](https://customers.pyimagesearch.com/lesson-sample-segmenting-characters-from-license-plates/)

[3] [http://tc11.cvc.uab.es/datasets/AcTiV_1](http://tc11.cvc.uab.es/datasets/AcTiV_1)

[4] [https://rrc.cvc.uab.es/?ch=4](https://rrc.cvc.uab.es/?ch=4)

[5] [https://rrc.cvc.uab.es/?ch=5&com=tasks](https://rrc.cvc.uab.es/?ch=5&com=tasks)

[6] [keras_ocr Documentation Fine-tuning the detector ](https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_detector.html)

[7] [keras_ocr Dataset.py Github](https://github.com/faustomorales/keras-ocr/blob/master/keras_ocr/datasets.py)

[8] [https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard](https://www.inflearn.com/course/tensorflow-%EC%8B%A4%EC%A0%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%98%AC%EC%9D%B8%EC%9B%90/dashboard)




