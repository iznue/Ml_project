# 학습한 ml model을 적용함
import cv2
import mediapipe as mp
import numpy as np
import joblib 
# pkl 모델 사용을 위해 pip install joblib 진행
# pip install scikit-learn
import pandas as pd
import warnings # 불필요한 경고문구를 지움


warnings.filterwarnings('ignore')

model = joblib.load('face.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # face_mesh 예측 시 예측 정확도 지정 가능
    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # 캡처된 프레임이 변경되지 못하게 copy하여 사용하거나 image.flags.writeable를 false로 지정함

        result = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 특징점을 찾아 그림을 그림
        mp_drawing.draw_landmarks(image, 
                                  result.face_landmarks,
                                  mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
        
        try:
            face = result.face_landmarks.landmark
            face_list = []
            for temp in face:
                face_list.append([temp.x, temp.y, temp.z])
            face_row = list(np.array(face_list).flatten())

            X = pd.DataFrame([face_row])

            class_name = ['happy', 'sad']
            yhat = model.predict(X)[0]
            yhat = class_name[yhat]
            # print(yhat)
            # print로 출력 시 실제 프레임 속도를 따라가지 못하므로 화면에 출력하는 것이 좋음

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, yhat, (30,30), font, 1, (255,0,0), 2)

        except:
            pass
        
        cv2.imshow('face', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
