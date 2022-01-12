import cv2, dlib, sys
import numpy as np

#scaler = 0.3

detector = dlib.get_frontal_face_detector() #얼굴 디텍터 모듈 초기화
predictor = dlib.shape_predictor(r'C:\Users\yongs\Downloads\shape_predictor_68_face_landmarks.dat') #얼굴 특징점 모듈 초기화
# shape_predictor은 이미 학습된 모델! -> 다운 받기

# 비디오 로드
cap = cv2.VideoCapture(0)
# 파일 이름 대신 0을 넣으 면 웹캠이 켜지고 내 얼굴로 테스트 가능

#오버레이 사진 로드
overlay = cv2.imread(r'C:\Users\yongs\Downloads\tory.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()

  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
  
  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
  b, g, r, a = cv2.split(img_to_overlay_t)
  
  mask = cv2.medianBlur(a, 5)
  
  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
  
  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
  
  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
  # convert 4 channels to 4 channels
  
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
  
  return bg_img

face_roi = []
face_sizes = []

while True:
    ret, img = cap.read()
    if not ret:
        break #영상이 없으면 종료
    
    ori = img.copy()

    #얼굴 인식 코드
    faces = detector(img) #img에서 모든 얼굴 찾기

    if len(face_roi) == 0:
        faces = detector(img, 1)
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        faces = detector(roi_img)
    
    if len(faces) == 0:
        print('no faces')
    
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(img, face) #img의 face 영역 안의 얼굴 특징점 찾기
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()]) #dlib 객체를 numpy 객체로 변환
        else:
            dlib_shape = predictor(roi_img, face) #img의 face 영역 안의 얼굴 특징점 찾기
            shape_2d = np.array([[p.x+face_roi[2], p.y+face_roi[0]] for p in dlib_shape.parts()]) #dlib 객체를 numpy 객체로 변환
    
        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        center_x, center_y = np.mean(shape_2d, axis = 0).astype(np.int)

    #얼굴 중앙점 계산
    top_left = np.min(shape_2d, axis=0) #최솟값 찾기
    bottom_right = np.max(shape_2d, axis=0) #최댓값 구하기

    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    
    face_size = max(bottom_right-top_left)
    face_sizes.append(face_size)

    if len(face_sizes) > 10:
        del face_sizes[0]

    mean_face_size = int(np.mean(face_sizes) * 1.8)

    face_roi = np.array([int(top_left[1] - face_size / 2), int(bottom_right[1] + face_size / 2), int(top_left[0] - face_size / 2), int(bottom_right[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    result = overlay_transparent(ori, overlay, center_x+8, center_y-25, overlay_size =(face_size, face_size))

  
    #이미지 영상 resize 하는 코드 (일단 나는 너무 작아서 , 안 씀)
    #img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))

    cv2.imshow('original',ori)
    cv2.imshow('img', img) #동영상을 읽어서 img라고 하는 윈도우에 띄우기
    cv2.imshow('result', result)
    
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1) 
        #1밀리세컨드만큼 대기 -> 이걸 넣어야 동영상이 제대로 보임!
 
