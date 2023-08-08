# 모자이크 작업 -> 얼굴로 검출된 영역을 축소하여 해상도를 낮춰 진행함

import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaseMesh = mp.solutions.face_mesh
faceMesh = mpFaseMesh.FaceMesh()

while True:
    ret, img = cap.read()
    # img : bgr이므로 face_mesh 사용을 위해서는 rgb로 변환해야 함
    imgRGB = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)

    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            xy_point = []
            for c,lm in enumerate(faceLms.landmark):
                xy_point.append([lm.x, lm.y])
        
        # print(xy_point)
        # print(len(xy_point))
        top_left = np.min(xy_point, axis=0)
        bottom_right = np.max(xy_point, axis=0)
        mean_xy = np.mean(xy_point, axis=0)

    ih, iw, ic = img.shape

    face_width = int(bottom_right[0]*iw) - int(top_left[0]*iw)
    face_height = int(bottom_right[1]*iw) - int(top_left[1]*iw)

    start_x = int(top_left[0]*iw)
    start_y = int(top_left[1]*ih)

    # 실제 얼굴 영역
    roi = img[start_y:start_y+face_height, start_x:start_x+face_width]
    roi = cv2.resize(roi, (int(face_width/10), int(face_height/10)))
    roi = cv2.resize(roi, (face_width, face_height), interpolation=cv2.INTER_AREA) # 확장한 것을 보관하기 위해 interpolation 지정

    try:
        img[start_y:start_y+face_height, start_x:start_x+face_width] = roi
        # roi 축소한 것을 그대로 적용함
    except:
        pass

    cv2.imshow('face', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()