from PIL import Image
import numpy
import pandas
import glob

# 현재 디렉토리 기준 data폴더 내의 모든 png파일 읽어오기

files = glob.glob('data/*.png')
pixels = []
emotion = []
usage = []
count = 0

fer2013new = pandas.read_csv('fer2013new.csv')
imageNameList = list(fer2013new['Image name'])

i = 0
for file in files:
    # 파일내의 이름 추출
    filename = file[5:]
    if filename in imageNameList :
        im = Image.open(file)

        # 해당 이미지 파일의 pixel값 추출
        pixels_temp = numpy.array(im).tolist()
        pixels_tempList = []
        for j in pixels_temp:
            pixels_tempList.append(str(j)[1:-1])
        pixels_temp = str(pixels_tempList).replace("'", "")
        pixels.append(pixels_temp)

        while(True):
            if filename == fer2013new.iloc[i]['Image name']:
                usage.append(fer2013new.iloc[i]['Usage'])

                # 해당 이미지 파일의 감정 추출(가장 많이 투표받은 것 기준 우선 순위)
                temp = [fer2013new.iloc[i]['neutral'], fer2013new.iloc[i]['happiness'], fer2013new.iloc[i]['surprise'],
                fer2013new.iloc[i]['sadness'], fer2013new.iloc[i]['anger'], fer2013new.iloc[i]['disgust'],
                fer2013new.iloc[i]['fear'],fer2013new.iloc[i]['contempt'], fer2013new.iloc[i]['unknown'],fer2013new.iloc[i]['NF']]
                
                emotion_temp = temp.index(max(temp))
                if emotion_temp == 0 :
                    emotion.append('neutral')
                    break
                elif emotion_temp == 1:
                    emotion.append('happiness')
                    break
                elif emotion_temp == 2:
                    emotion.append('surprise')
                    break
                elif emotion_temp == 3:
                    emotion.append('sadness')
                    break
                elif emotion_temp == 4:
                    emotion.append('anger')
                    break
                elif emotion_temp == 5:
                    emotion.append('disgust')
                    break
                elif emotion_temp == 6:
                    emotion.append('fear')
                    break
                elif emotion_temp == 7:
                    emotion.append('contempt')
                    break
                elif emotion_temp == 8:
                    emotion.append('unknown')
                    break
                else :
                    emotion.append('NF')
                    break
            else:
                i+=1
    else:
        continue

fer2013plus = pandas.DataFrame(list(zip(emotion, pixels, usage)), columns=(['emotion', 'pixels', 'Usage']))
for i in range(len(fer2013plus)) :
    fer2013plus.iloc[i]['pixels'] = fer2013plus.iloc[i]['pixels'][1:-1].replace(',', '')

fer2013plus.to_csv('fer2013plus.csv', sep=',', header=True, index=False)