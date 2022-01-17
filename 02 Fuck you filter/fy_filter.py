import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 2 # 인식되는 손의 수(default = 2)
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy',
    12:'love', 13:'이', 14:'지', 15:'원'
} #여기서 fy 클래스 정의 -> fuck you 표시

# MediaPipe hands model
mp_hands = mp.solutions.hands #손 인식
mp_drawing = mp.solutions.drawing_utils # ???
hands = mp_hands.Hands(
    max_num_hands=max_num_hands, 
    min_detection_confidence=0.5,#최소 신뢰구간 of detection (default=0.5)
    min_tracking_confidence=0.5) #최소 신뢰구간 of tracking (default=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train_love.csv', delimiter=',') #데이터를 불러오기 using python -> numpy 필요
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32) # 데이터 타입 지정 -> 외우기
knn = cv2.ml.KNearest_create() #knn 모델 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # (train, cv2.ml.ROW_SAMPLE, response)

#웹캠 켜기
cap = cv2.VideoCapture(0)

#클릭할 때마다 데이터 저장
def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1) # 좌우 및 상하 반전: 1은 좌우, 0은 상하
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #색상 공간 변환: BGR -> RGB

    result = hands.process(img) #이미지 처리 

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None: #hands의 landmarks 모음 -> landmarks가 없으면?
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3)) # 빨간 점이 joint
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] # 직접 설정

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            # data = np.append(data, 12) 데이터 수집 코드
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            
            if idx == 11:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img

            elif True:
                cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

np.savetxt('gesture_train_love.csv', file, delimiter=',')