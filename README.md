## Team Name: slow_starter
- Member: 양한수, 김규미

## 프로젝트 목표
- 실제 cctv 영상파일을 다운받아 사진화 하여 차량인식 테스트(차, 트럭, 버스)
- 프로젝트 진행 과정에서 발생하는 에로사항 관찰 및 해결방식 학습

## 데이터 출처
- Ai-Hub: 교통문제 해결을 위한 CCTV 교통 영상(고속도로) 파일 사용 [(출처)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=164)
![데이터 출처](https://github.com/sesac-google-ai-1st/slow_starter_repo/assets/147117477/a61776ab-48d9-4881-8a86-7fa1ae50609b)

- 데이터 용량이 너무 커서(약 90GB), 각 train,val 데이터에서 CH01~04번 데이터(약 30GB)까지만 활용하기로 함

## 프로젝트 개발환경
- PyTorch:2.0
- Ultralytics YOLOv8.0.216
- Python-3.10.13

## 작업 과정

### 00. 사전 작업 (참고: highway_project.ipynb )

1) 데이터셋 파일 다운받은 후 신규 버킷 생성하여 업로드 진행
   - 버킷 생성 시 (멀티리전/단일리전) 중 단일리전 으로 버킷 생성함(다른 리전에 접근할 경우가 없으므로 속도를 위해 선택)
   - 대용량 파일이므로 한번에 업로드 중 네트워크 오류 발생시 다시 처음부터 업로드를 해야하는 위험성이 잇음
   - 반디집을 이용해 분할 압축 후 버킷에 업로드 하여 테스트 진행 시 재압축하는식으로 하는 방식도 가능(문제 발생시 재업로드 편의성 증가)

### 01. 데이터셋 준비하기
- 버킷과 gcp로 실행한 jpyter notebook 연동
  - gcsfuse 로 마운트(colab의 드라이브 마운트와 같이 실시간 연동하는 식)
  - gsutill 로 접속하여 복사(cp) 하거나 다운로드를 통해 jpyter lab의 local로 옮겨서 사용(원본은 변화안됨, 딴사람의 버킷에 접속하여 다운가능)
  - gsutill 로 연동하면 복사/업로드 과정이 gcsfuse 보다 2배 이상 빠름
- 가져온 압축파일 압축풀기


### 01-1. 앞에서 생성한 dataset폴더에 버킷 마운트하기  
```
BUCKET_NAME = 'yolostudy'
MOUNT_PATH ='/home/jupyter/dataset' 
!gcsfuse --implicit-dirs {BUCKET_NAME} {MOUNT_PATH}
```

### 01.2. 압축풀기 









