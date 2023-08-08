import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaseMesh = mp.solutions.face_mesh
faceMesh = mpFaseMesh.FaceMesh()

faceimg = cv2.imread('face_mk.png', cv2.IMREAD_UNCHANGED)

def change_mask(background_img, img_to_overlay, x, y, overlay_size=None):
    # 무조건 try, except로 예외처리 후 작업
    # background_img : 얼굴 영역 크기
    # 함수에서 처리하는 동안 얼굴 각도가 달라지면 이상이 생길 수 있으니 background_img를 copy하여 사용
    try:
        bg_img = background_img.copy()

        if bg_img.shape[2] == 3: # 3채널인 경우
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
            # face_mk의 경우 png이기 때문에 4채널이므로 채널을 맞추기 위해 해당 작업을 거침
        
        if overlay_size is not None: # 마스크 확대 값이 주어지는 경우, face 이미지 사이즈가 재설정 되었으면 재설정된 이미지로 변경함
            img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)
    
        b,g,r,a = cv2.split(img_to_overlay) # 분리 후 alpha 채널만 사용(투명)
        # mask 사용 시 블러처리를 하는 것이 좋음
        mask = cv2.medianBlur(a, 5)
        
        h,w,_ = img_to_overlay.shape

        i_s = int(y - h / 2)
        i_e = int(y + h / 2)
        c_s = int(x - w / 2)
        c_e = int(x + w / 2)
        if i_s < 0:
            i_s = 0
        if i_e > bg_img.shape[0]:
            i_e = bg_img.shape[0]
        if c_s < 0:
            c_s = 0
        if c_e > bg_img.shape[1]:
            c_e = bg_img.shape[1]
        roi = bg_img[i_s:i_e,c_s :c_e]  # 실제 얼굴에서 mask 영역을 빼줌

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask)) # bit 연산 진행
        # mask가 씌워진 부분을 제외한 영역을 따로 저장함
        img2_fg = cv2.bitwise_and(img_to_overlay, img_to_overlay, mask=mask)

        # 캠이미지에 마스크영역과 마스크를 제외해서 더한 결과값을 갱신
        bg_img[i_s:i_e,c_s :c_e] = cv2.add(img1_bg, img2_fg)
        
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img
    
    except:
        return background_img

############################################################################################################

while True:
    ret, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)

    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            xy_point = []
            for c,lm in enumerate(faceLms.landmark):
                ih,iw,ic = img.shape

                img = cv2.circle(img, (int(lm.x*iw), int(lm.y*ih)), 1, (255,0,0), 3)

                xy_point.append([lm.x, lm.y])

            top_left = np.min(xy_point, axis=0)
            bottom_right = np.max(xy_point, axis=0)
            mean_xy = np.mean(xy_point, axis=0)

            face_width = int(bottom_right[0]*iw) - int(top_left[0]*iw)
            face_height = int(bottom_right[1]*iw) - int(top_left[1]*iw)

            if face_width > 0 and face_height > 0:
                result = change_mask(img, faceimg, int(mean_xy[0]*iw), int(mean_xy[1]*ih))

    try:
        cv2.imshow('image', result)
    except:
        cv2.imshow('image',img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
