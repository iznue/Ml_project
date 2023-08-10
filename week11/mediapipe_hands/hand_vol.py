import cv2
import mediapipe as mp
import numpy as np
import math
import socket
# socket : 실시간 통신에 사용

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# socket : udp 통신 사용
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sendport = ('127.0.0.1', 5053) # 상대방 서버 : 주소와 포트번호 작성
# 최종적으로는 상대 서버에 vol 값을 전송함

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = cap.read()
        image = cv2.flip(image, 1) # image 좌우반전 -> 손 인식의 경우 좌우반전을 적용해야 헷갈리지 않음
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 검출된 손 데이터가 있는 경우
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                # print('----------------------------------')

                p1 = hand_landmarks.landmark[4]
                p2 = hand_landmarks.landmark[8]

                # 두 손 좌표의 거리 값 구하기
                a = p1.x - p2.x
                b = p1.y - p2.y
                c = math.sqrt((a*a)+(b*b))
                vol = int(c*100)
                vol = np.abs(vol)

                ################################################################
                # 상대 서버에 보낼 때는 숫자가 아닌 문자로 전송함
                senddata = str(vol)
                sock.sendto(str.encode(senddata), sendport)
                ################################################################

                cv2.putText(image, text='Volume : %d'%vol,
                            org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=255, thickness=2)
        
                mp_drawing.draw_landmarks(image,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)
                
                
        cv2.imshow('hand', image)

        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()