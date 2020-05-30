#### 용량이 크기 때문에 fer2013.csv 데이터는 Kaggle에서 직접 다운받아 주시길 바랍니다.

해당 데이터는 FER2013(Facial Emotion Recognition 2013)의 데이터셋을 크라우드소싱을 통해 10명의 Tagger들이 투표를 진행하여 가장 많이 선택된 감정을 기반으로 만든 데이터셋입니다.
기존 FER2013 데이터에서 라벨링이 잘못된 데이터들로 인한 낮은 Accuracy 예방할 수 있습니다.


* 출처 : [FERPlus](https://github.com/Microsoft/FERPlus)

---

1. src 폴더에 존재하는 generate_training_data.py를 통해 image 파일들을 생성합니다.
```
python generate_training_data.py -d <dataset base folder> -fer <fer2013.csv path> -ferplus <fer2013new.csv path>
```
2. 생성된 이미지 파일들을 data 폴더로 모아줍니다. label.csv 파일은 지우셔도 무방합니다. 아래와 같이 환경을 구성합니다.
```
/data
    fer0000000.png ~ fer0035801.png
/src
    #6 py files
README.md
data2csv.py
fer2013new.csv
```
3. data2csv.py 파일을 실행시키면 fer2013plus.csv 데이터를 생성할 수 있습니다.
