import argparse
import econ
import socket
import numpy as np
import cv2
from imutils import face_utils
import dlib
import torch
from PIL import Image
from eyeconModel import FERModel, GazeModel

IMG_WIDTH , IMG_HEIGHT= 480, 640
ANRD_WIDTH,ANRD_HEIGHT = 1170,1780


device = torch.device('cpu')

emotion_dict = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger",
5: "disgust", 6: "fear", 7: "comtempt", 8: "unknown", 9: "Not a Face"}


parser = argparse.ArgumentParser()
parser.add_argument('--util_path',default='./utils/',required = False, help='util file path')
parser.add_argument('--blink_th',default=0.25,required=False, help='eye blick value threshold')
parser.add_argument('--port',default=8200,type=int)
parser.add_argument('--ip',default='192.168.0.25')

args = parser.parse_args()

util_path  = args.util_path
IP,PORT = args.ip, args.port
BLINK_TH = args.blink_th



faceClassifier = econ.loadClassifier(util_path)

predictor = dlib.shape_predictor(util_path + 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


districtList = econ.getDistrictList(ANRD_WIDTH,ANRD_HEIGHT)


d_class = 16

d_class_count = 4
state_direction_num = 5
state_Q_num = 3

state_memory = econ.loadStateDict(d_class,d_class_count,state_direction_num,state_Q_num)

Gaze_model = GazeModel()
Gaze_model.eval()

FER_model= FERModel()
FER_model.eval()



print(IP,':',PORT)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen()
print('listening...')


try:
    client_socket, addr = server_socket.accept()
    print('Connected with ', addr)

    count = 0
        
    while True:
        print('Receive IMG',end='\t| ')

        length = econ.recvAll(client_socket, 10)
        frame_bytes = econ.recvAll(client_socket, int(length))
        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = image[50:IMG_HEIGHT-50][50:IMG_WIDTH-50]

        print('Transform IMG',end='\t| ')

        # frame = imutils.resize(image, width=450)





        print('Gaze tracking [Classification]', end='\t| ')


        inputData = Image.fromarray(image)
        inputData = econ.transformImg(inputData)
        inputData = torch.autograd.Variable(inputData, requires_grad = True)

        state_memory['District'][count % d_class_count] =  econ.getGazeDistrictIdx(Gaze_model, inputData)
        gazeIdx = econ.modeList(state_memory['District'])
        gazePoint = districtList[gazeIdx]


        print('Gaze tracking [Gaze Ratio]', end='\t| ')

        rects = detector(gray, 0)
        for rect in rects:
            faceLandmark = predictor(gray, rect)

        if(len(rects) != 0) :
            horizontal_gaze_ratio_left_eye, vertical_gaze_ratio_left_eye =  econ.getGazeRatio(image,gray, faceLandmark,np.arange(lStart,lEnd+1))
            horizontal_gaze_ratio_right_eye, vertical_gaze_ratio_right_eye =  econ.getGazeRatio(image,gray, faceLandmark,np.arange(rStart,rEnd+1))
            horizontal_gaze_ratio = (horizontal_gaze_ratio_right_eye + horizontal_gaze_ratio_left_eye) / 2
            vertical_gaze_ratio = (vertical_gaze_ratio_right_eye + vertical_gaze_ratio_left_eye) / 2

            state_memory['GazeRatioLR'][count % state_direction_num] = horizontal_gaze_ratio
            state_memory['GazeRatioTB'][count % state_direction_num] = vertical_gaze_ratio

            gazeRatioLR = horizontal_gaze_ratio - state_memory['GazeRatioLR'].mean()
            gazeRatioTB = vertical_gaze_ratio - state_memory['GazeRatioTB'].mean()
        else :
            state_memory['GazeRatioLR'][(count) % state_direction_num] = 1
            state_memory['GazeRatioTB'][(count) % state_direction_num] = 1

        gazeRatioLR = state_memory['GazeRatioLR'][(count) % state_direction_num] - state_memory['GazeRatioLR'].mean()
        gazeRatioTB = state_memory['GazeRatioTB'][(count) % state_direction_num] - state_memory['GazeRatioTB'].mean()

        print('Gaze tracking [Face Point]')

        facePoints = econ.classifyFace(image, faceClassifier)

        if(len(facePoints)!= 0 ) :
            faceX,faceY = econ.getFaceXY(facePoints)
            state_memory['FacePointX'][count % state_direction_num],\
            state_memory['FacePointY'][count % state_direction_num] = faceX,faceY
        else :
            state_memory['FacePointX'][count % state_direction_num] = state_memory['FacePointX'][(count-1) % state_direction_num]
            state_memory['FacePointY'][count % state_direction_num] = state_memory['FacePointY'][(count-1) % state_direction_num]

        
        faceDirectionX = (state_memory['FacePointX'][count%state_direction_num] - state_memory['FacePointX'].mean())/IMG_WIDTH
        faceDirectionY = (state_memory['FacePointY'][count%state_direction_num] - state_memory['FacePointY'].mean())/IMG_HEIGHT


        print('Check blink',end='\t| ')

        if(len(rects) != 0 ) :
            eyeLandmark = face_utils.shape_to_np(faceLandmark)

            if(econ.isBlink(eyeLandmark,BLINK_TH,lStart,lEnd)) :
                state_memory['Click'][count%state_Q_num] = 0
            else :
                state_memory['Click'][count % state_Q_num] = 1

            if(econ.isBlink(eyeLandmark,BLINK_TH,rStart,rEnd)) :
                state_memory['Scroll'][count % state_Q_num] = 0
            else :
                state_memory['Scroll'][count % state_Q_num] = 1
        else :
            state_memory['Click'][count % state_Q_num] =0
            state_memory['Scroll'][count % state_Q_num] =0


        print('Estimate expression',end='\t| ')


        state_memory['FER'][count%state_Q_num], tagPosition = econ.getExpression(facePoints,image,FER_model)



        blink = econ.modeList(state_memory['Click'])
        scroll = econ.modeList(state_memory['Scroll'])
        if (blink == 1 or scroll == 1):
            FERNum = 0
            tagPosition = (0, 0)
        else:
            FERNum = econ.modeList(state_memory['FER'])
            cv2.putText(image,emotion_dict[FERNum],tagPosition, cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0), 1, cv2.LINE_AA)


        print('Sent to Client',end='\t')


        # print(gazeRatioLR,gazeRatioTB,'\n',faceDirectionX,faceDirectionY)
        # x, y  = econ.getXY(cur_point, facePoint,count,diff_TH,freeze_TH)
        # x, y  = econ.strechingPoint(x,y,IMG_WIDTH,IMG_HEIGHT,ANRD_WIDTH,ANRD_HEIGHT)

        x,y = gazePoint
        x,y =  int(ANRD_WIDTH/2), int(ANRD_HEIGHT/2)
        device_width = ANRD_WIDTH / 4
        device_height = ANRD_HEIGHT / 4
        d1_x,d1_y = econ.rateToDistance(gazeRatioLR,gazeRatioTB,device_width,device_height,weight=5)
        d2_x, d2_y = econ.rateToDistance(faceDirectionX, faceDirectionY, device_width, device_height,weight=5)
        x += d1_x +d2_x
        y += d2_x +d2_y
        # gazeRatioLR
        cord = str(x) + '/' + str(y) + '/' +str(blink) + '/' + str(scroll) + '/' + str(FERNum)
        print(cord)
        client_socket.sendall(bytes(cord,'utf8'))



        cv2.imshow('Android Screen', image)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as e:
    print(e)
    print('Connecting Error')
    pass
finally:
    print('End of the session')


