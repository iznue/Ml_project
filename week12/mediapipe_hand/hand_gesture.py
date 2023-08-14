import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

max_num_hands = 1
gesture = {0:'rock', 1:'paper', 2:'scissors'}
# gesture = {0:'stop', 1:'fire'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

total_result = []

# 마우스를 클릭하는 경우 데이터를 저장함
def click(event,x,y,flags,params):
    global data
    if event == cv2.EVENT_LBUTTONDOWN:
        print('mouse Click')
        total_result.append(data)
        print(data)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

with mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None: 
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3)) # 각 인덱스 21개는 좌표를 3개씩 가지므로 다음과 같이 지정
                for j,lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # joint의 index 순서대로 가져옴
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], : ] 
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], : ] 
                v = v2 - v1

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 정규화 작업

                angle = np.arccos(np.einsum('nt, nt->n',
                                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], : ],
                                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], : ])) # Einstein summation notation 사용, radian 값이 구해짐
                
                angle = np.degrees(angle) # radian -> degree

                data = np.array([angle], dtype=np.float32)
                data = np.append(data, 2) # class 지정, data array 가장 뒤에 각 레이블들이 저장됨

                mp_drawing.draw_landmarks(img, res,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            

        cv2.imshow('Dataset', img)
        # 윈도우의 이름을 전부 동일하게 맞춰줘야 함

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

total_result = np.array(total_result, dtype=np.float32)
df = pd.DataFrame(total_result)
df.to_csv('hand.csv', mode='a', index=None, header=None) # mode를 여러개로 지정할 예정이므로 'a'로 설정
