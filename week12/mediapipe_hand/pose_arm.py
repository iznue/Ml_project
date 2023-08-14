############# 왼쪽 어깨(11,13,15번) 각도 구하기

import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# 3점 사이 각도 구하는 함수
def three_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        result = pose.process(image)

        image.flags.writeable = True # lock 작업

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image,
                                  result.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

        try:
            landmarks = result.pose_landmarks.landmark
            # 관절 명칭에 따라 찾기, x,y,z 좌표 값을 담음
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            
            v_s_e = np.array(elbow) - np.array(shoulder)
            v_e_w = np.array(elbow) - np.array(wrist)
            # print(v_s_e)

            norm_s_e = np.linalg.norm(v_s_e)
            norm_e_w = np.linalg.norm(v_e_w)

            dot_s_w = np.dot(v_s_e, v_e_w)

            cos_th = dot_s_w / (norm_s_e * norm_e_w)
            rad = math.acos(cos_th)
            deg = math.degrees(rad)
            # print(deg)

            print(three_angle(shoulder, elbow, wrist))
            
        except:
            pass

        cv2.imshow('pose', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
