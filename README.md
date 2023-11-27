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
- 구글 GCP 사용(PyTorch:2.0, GPU: NVIDIA V100 * 2, CPU Core: 16)
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

=====================================================================================

### 01-1. 앞에서 생성한 dataset폴더에 버킷 마운트하기  
```
BUCKET_NAME = 'yolostudy'
MOUNT_PATH ='/home/jupyter/dataset' 
!gcsfuse --implicit-dirs {BUCKET_NAME} {MOUNT_PATH}
```

=====================================================================================

### 01.2. 압축풀기 
- 여러가지 압축푸는 방식 존재, 분할압축하여 버킷에 업로드 했다면 분할된 압축파일을 다시 재압축 후 unzip 해야함
```
!zip -s 0 image.zip --out iamge_total.zip
```

1) unzip [압축파일] -d [압축 풀 경로]
   - 압축 푸는 시간이 오래걸림, 약 1시간 30분 이상...
  
2) 현재 컴퓨터의 가용 CPU Core를 전부 사용하여 압축을 푸는 multiprocessing 기능 사용
   - 압축푸는 시간 크게 단축됨.

```
%%time
%cd ~/

import os
import zipfile
import multiprocessing
import concurrent.futures

cpu_num = os.cpu_count()  # 현재 pc의 cpu 코어 수 확인

## [라벨]1.수도권영동선 - 16MB = 1.07 s
# zipfilePath = 'highway/Training/bounding_box/[라벨]1.수도권영동선.zip'
# savePath = 'highway/Training/bounding_box/labels_total'

## [원천]1-1.수도권영동선 - 29.8GB = 56min 21s(cpu = 4) => 17min 7s(cpu = 14)
# zipfilePath = 'highway/Training/bounding_box/[원천]1-1.수도권영동선.zip'
# savePath = 'highway/Training/bounding_box/images_total'

## [라벨]1.수도권영동선 - 2.5MB = 6.27 s(cpu = 4) => 463 ms(cpu = 14)
# zipfilePath = 'highway/Validation/bounding_box/[라벨]1.수도권영동선.zip'
# savePath = 'highway/Validation/bounding_box/labels_total'

## [원천]1.수도권영동선 - 9GB =  18min 31s(cpu = 4) => 3min 31s(cpu = 14) 
## - 폴더 이름 깨짐 이슈때문에 jupyter local로 압축파일 복사하여 압축풀기 후 mv로 폴더이름 변경 진행함
# zipfilePath = 'highway/val/[원천]1.수도권영동선.zip'
# savePath = 'highway/val/images_total'

def unzip(file):
    with lock:
        zf.extract(file, path=savePath)

zf = zipfile.ZipFile(zipfilePath)    

m = multiprocessing.Manager()
lock = m.Lock()  # 여러개의 폴더의 압축을 풀떄 하나의 폴더의 압축을 다 풀때까지 다른작업 중지하기

with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num-2) as executor:
    executor.map(unzip, zf.infolist())  # map = 여러개의 기능을 동시에 실행
```

=====================================================================================

### 02. 데이터 전처리
- 전처리 중 발생한 이슈 확인 및 해결

### 02-1. train/val 폴더를 생성하여 png 파일 전부 몰아넣기
- 학습의 사용될 YOLO 모델의 형식대로 데이터 정리가 필요함
- train,val 폴더 아래에 각각 images, labels 폴더 아래에 데이터가 위치해야함
- 기존 구성에서 iamges 폴더 아래에 여러개의 폴더에 나뉘어서 데이터가 위치하였으므로 각각의 폴더 내부의 접근하여 데이터를 옮기는 함수를 작성하여 실행
- 함수 선언(2중 for문형태의 코드를 간편화 하기 위해 내부 for문을 함수로 지정하여 따로 선언)
```
# images_total 하위의 폴더명 리스트를 전달받은 후,
# 해당 폴더 아래에 접근하여 폴더 내에 존재하는 이미지 파일들을 목적지(dstDir)로 옮기는 함수 선언
# 전달받을 parameter:
##  baseDir: /highway/val/images_total/폴더명
##  srcFiles: images_total/폴더/ 아래에 있는 이미지 파일들의 리스트
##  dstDir: 데이터를 이동시킬 최종 목적지 highway/val/images

def move_files(baseDir, srcFiles, dstDir):
    for srcFile in srcFiles:
        srcFilePath = os.path.join(baseDir, srcFile)  # 각각의 이미지데이터(png,jpg)의 전체 경로 저장
        #print(srcFilePath)
        #print(srcFilePath, dstDir)
        shutil.move(srcFilePath, dstDir) # 경로가 저장된 각각의 이미지파일을 롬기는 명령 실행
```
- 함수 호출(images 폴더 아래에 각 폴더에 접근할떄마다 해당 폴더 아래의 데이터를 옮기는 코드 실행)
```
%%time
## image파일만 옮기기 실행, label 데이터는 xml파일을 txt파일로 변환하면서 저장 예정
## images_total 아래에 있는 각각의 폴더에 있는 png 파일을 전부 추출해 iamges폴더로 옮기기

## 초기 경로 지정

# val/iamges_total 데이터 옮기기 = 총 3333개
# srcBaseDir = os.path.join(dataPath, "highway/val/images_total")
# destPath = os.path.join(dataPath,"highway/val/images")

# tarin/images_total 데이터 옮기기 = 총 24102개
# srcBaseDir = os.path.join(dataPath, "highway/train/images_total")
# destPath = os.path.join(dataPath,"highway/train/images")


## =========== txt_total에 있는 txt 파일을 옮기기 전에 xnkTotxt 변환작업을 먼저 수행해야 함 ============ ##
## =========== 레이블xml2txt변환하기.ipynb 파일 참조 ============ ##

# val/txt_total 데이터 옮기기 = 총 3334 개
# srcBaseDir = os.path.join(dataPath, "highway/val/txt_total")
# destPath = os.path.join(dataPath,"highway/val/labels")

# train/txt_total 데이터 옮기기 = 총 24104개
#srcBaseDir = os.path.join(dataPath, "highway/train/txt_total")
#destPath = os.path.join(dataPath,"highway/train/labels")



# images_total 하위의 디렉토리 목록을 리스트 데이터로 생성
dirNames = os.listdir(srcBaseDir)

# 총 이동시킨 file의 개수를 확인하기 위해 사용
fileNumCounter=0

# images_total 하위의 디렉토리 목록 리스트를 하나씩 꺼내어 반복문 실행
for dirName in dirNames:
    srcBasePath = os.path.join(srcBaseDir,dirName)  # images_total/dirname경로 저장
    fileNames = os.listdir(srcBasePath)  # images_total/dirname경로 를 사용하여 images_total/dirname 폴더 내부의 이미지 파일목록을 리스트로 저장
    print("dir:{}, fileNum:{}".format(dirName, len(fileNames)))  # 폴더 이름과 각 폴더별 내부 이미지 데이터 개수 확인을 위해 출력
    fileNumCounter+=len(fileNames)  # 각 폴더의 내부 데이터 개수를 합산하기 위해 저장
    
    # 각 폴더명에 접근하여 내부 데이터를 옮기는 함수 호출
    move_files(srcBasePath, fileNames, destPath)
    #print(fileNames)

# 총 옮겨진 파일 개수 확인
print("fileNumCounter:{}".format(fileNumCounter))
```
- 실행 결과
```
dir:Suwon_CH02_20200720_2130_MON_9m_NH_highway_TW5_sunny_FHD, fileNum:150
dir:Suwon_CH04_20201012_1838_MON_9m_RH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH04_20200721_1830_TUE_9m_RH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH01_20201012_1723_MON_9m_RH_highway_TW5_sunny_FHD, fileNum:150
dir:Suwon_CH03_20201012_1933_MON_9m_RH_highway_OW5_sunny_FHD, fileNum:149
dir:Suwon_CH02_20200722_1730_WED_9m_NH_highway_TW5_sunny_FHD, fileNum:50
dir:Suwon_CH01_20200721_1700_TUE_9m_RH_highway_TW5_sunny_FHD, fileNum:150
dir:Suwon_CH04_20200720_1830_MON_9m_NH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH04_20201010_1818_SAT_9m_RH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH02_20200721_2030_TUE_9m_NH_highway_TW5_sunny_FHD, fileNum:100
dir:Suwon_CH03_20201011_1742_SUN_9m_RH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH04_20200722_1600_WED_9m_NH_highway_OW5_rainy_FHD, fileNum:625
dir:Suwon_CH03_20200721_2000_TUE_9m_NH_highway_OW5_sunny_FHD, fileNum:100
dir:Suwon_CH03_20200722_1700_WED_9m_NH_highway_OW5_sunny_FHD, fileNum:100
dir:Suwon_CH01_20200722_1930_WED_9m_NH_highway_TW5_sunny_FHD, fileNum:50
dir:Suwon_CH03_20200720_2030_MON_9m_NH_highway_OW5_sunny_FHD, fileNum:150
dir:Suwon_CH01_20200722_1430_WED_9m_NH_highway_TW5_rainy_FHD, fileNum:50
dir:Suwon_CH02_20201011_1806_SUN_9m_RH_highway_TW5_sunny_FHD, fileNum:150
dir:Suwon_CH01_20200720_1830_MON_9m_RH_highway_TW5_sunny_FHD, fileNum:150
dir:Suwon_CH01_20201213_1200_SUN_9m_NH_highway_TW5_snow_FHD, fileNum:230
dir:Suwon_CH02_20201213_0933_SUN_9m_NH_highway_TW5_snow_FHD, fileNum:230
fileNumCounter:3334
CPU times: user 50.2 ms, sys: 66.5 ms, total: 117 ms
Wall time: 141 ms
```

- 해당 작업 이후 데이터가 없는 빈 파일들 삭제 진행

=====================================================================================

### 02-2. 파일명 일치 확인 (참조: 파일명일치확인(train).ipynb, 파일명일치확인(val).ipynb)
1) txt파일명이 잘못된 것 수정
   - train 파일의 images와 labels의 파일명이 일치하지 않는것 약 150개 발견
   - 이미지 이름은 'Suwon_CH02_20200721_2130_TUE_9m_NH_highway_TW5_sunny_FHD.png'
   - 텍스트 파일은 'Suwon_CH02_20200721_2130_TUE_9m_NH_highway_TW5_sunny_FHD 001.txt'
   - txt파일명에 언더바 대신 공백이 있는 것 확인
   - 아래 코드로 txt 파일명에 언더바(_) 추가 하여 해결
   ```
   falseLabels = glob('/home/jupyter/highway/train/labels/Suwon_CH02_20200721_2130*.txt')

   for falseLabel in falseLabels:
       newLabel = falseLabel.replace(' ','_')
       print(newLabel)
       os.rename(falseLabel, newLabel)
   ```
   
=====================================================================================

2) txt 파일명 수정 후에도 여전히 iamges, rabels 데이터 개수 불일치 확인
- labels 폴더에 txt파일이 3개 더 많은것을 확인한 후 결측치인지 확인하기
- labels 폴더에서 아래 3개의 파일이 images 파일에 없는것을 확인
   - Suwon_CH02_20200721_1530_TUE_9m_NH_highway_TW5_sunny_FHD_069.txt
   - uwon_CH03_20200722_1900_WED_9m_NH_highway_OW5_sunny_FHD_051.txt
   - Suwon_CH03_20200722_1900_WED_9m_NH_highway_OW5_sunny_FHD_052.txt
- 원본 압축파일에서 확인해본 결과 해당 파일이 없는것을 확인
- 결측치로 판단하여 labels 폴더에서 해당 텍스트 파일 삭제하는 코드 작성하여 실행
```
# split_labelfiles 에는 labels폴더 하위의 데이터리스트들의 이름이 담겨잇음 
true_count=0
false_count=0
for label in split_labelfiles:
    if label in split_imagefiles:
        true_count+=1
    else:
        false_count+=1
        print('불일치 파일: {}'.format(label))
        falsefilePath = os.path.join(labelfullPath,label + '.txt')
        #print(falsefilePath)
        #해당파일 삭제
        os.remove(falsefilePath)
        print('파일 삭제 완료!!!')
        

print(f'true_count:{true_count}')
print(f'false_count:{false_count}')
```
- val 데이터 폴더 아래에서도 한개의 label 파일이 많은것을 확인하여 차집합 계산식을 활용해 검출하여 삭제함
   - 결측 파일 검출 
   ```
   s1 = set(split_imagefiles)
   len(s1)
   s2 = set(split_labelfiles)
   len(s2)
   s2-s1
   ```

   ```
   {'Suwon_CH02_20200721_2030_TUE_9m_NH_highway_TW5_sunny_FHD_001'}
   ```
   - 결측 파일 삭제
  ```
   os.remove('/home/jupyter/highway/val/labels/Suwon_CH02_20200721_2030_TUE_9m_NH_highway_TW5_sunny_FHD_001.txt')
  ```
=====================================================================================

### 02-3. VOLO nano model test 중 train의 png 파일 하나를 읽지 못하는 메시지 발생한것 확인
- 'Suwon_CH02_20200721_2130_TUE_9m_NH_highway_TW5_sunny_FHD_096.png'
- highway/train/images/Suwon_CH02_20200722_1530_WED_9m_NH_highway_TW5_rainy_FHD_048.png 파일을 확인해보니 0KB인것 확인함
- 압축 풀기 전 원본 데이터 확인해보니 해당 파일 존재함
- 압축 푸는과정에서 손상난것으로 추정
- 손상파일 삭제 후, 원본 압축파일에서 해당 png 파일만 추출해서 업로드하는 식으로 파일 교체함


=====================================================================================


### 03. 모델 학습시키기 (참조: 고속도로CCTV데이터기반차량인식.ipynb)

1) 1차 테스트:nano model test - epochs=75 설정

![confusion_matrix_normalized](https://github.com/sesac-google-ai-1st/slow_starter_repo/assets/147117477/05cf1fe1-f500-4725-af60-c1fc2ec6ca4c)

![results](https://github.com/sesac-google-ai-1st/slow_starter_repo/assets/147117477/424e7f52-5379-4f4d-b2f4-a5bea0f6aef0)

### 결과 분석 -> mAP50-95 의 값이 0.76까지 균등하게 증가하였고 loss 값도 꾸준하게 감소하였다.
<br><br>

2) 2차 테스트: nano model test - epochs=150 설정

![confusion_matrix_normalized](https://github.com/sesac-google-ai-1st/slow_starter_repo/assets/147117477/4ba3186d-7e05-4707-bd8a-870f11ab367d)

![results](https://github.com/sesac-google-ai-1st/slow_starter_repo/assets/147117477/0f440302-c88e-4249-b3b1-e1fcf363e95f)


### 결과 분석 -> mAP50-95 의 값이 0.77까지 추가 상승한것을 확인함



## 시간적 한계로 모델 학습 테스트는 여기까지만 수행하였습니다.
## epochs 수를 좀더 늘리면 mAP50-95 값이 조금 도 올라갈 수도 있을 것이라 추정중입니다.















