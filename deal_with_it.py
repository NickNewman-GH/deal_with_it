import cv2
import numpy as np

def detect_faces(image, face_classifier):
    return face_classifier.detectMultiScale(image, 1.2, 5)

def detect_eyes(image, face_rects, eyes_classifier):
    eyes = []
    for i, (x,y,w,h) in enumerate(face_rects):
        eyes.append(eyes_classifier.detectMultiScale(image[y: y + h, x: x + w], 1.2, 5))
        face_eyes_count = len(eyes[i])
        if face_eyes_count > 0:
            for j in range(face_eyes_count):
                eyes[i][j][0] += x
                eyes[i][j][1] += y
    return eyes

def delete_borders(image):
    min_index_y, min_index_x = 0, 0
    max_index_y, max_index_x = image.shape[0] - 1, image.shape[1] - 1
    for i in range(image.shape[0]):
        if np.max(image[i, :, 3]) > 0:
            min_index_y = i
            break
    for i in range(image.shape[1]):
        if np.max(image[:, i, 3]) > 0:
            min_index_x = i
            break
    for i in range(image.shape[0] - 1, -1, -1):
        if np.max(image[i, :, 3]) > 0:
            max_index_y = i
            break
    for i in range(image.shape[1] - 1, -1, -1):
        if np.max(image[:, i, 3]) > 0:
            max_index_x = i
            break
    image = image[min_index_y:max_index_y + 1,min_index_x:max_index_x + 1]
    return image


def draw_glasses(image, eye_rects, glasses_image):
    for eyes in eye_rects:
        if len(eyes) > 1:
            eyes = sorted(eyes, key = lambda eye: eye[0])
            left_eye = eyes[0]
            right_eye = eyes[-1]

            med_width = (right_eye[2] + left_eye[2])//2
            glasses_width = right_eye[0] - left_eye[0] + right_eye[2] + med_width
            glasses_height = int(glasses_image.shape[0] / (glasses_image.shape[1]/glasses_width))

            glasses_image = cv2.resize(glasses_image, (glasses_width, glasses_height))
            new_image = np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
            new_image[min(left_eye[1],right_eye[1])+glasses_image.shape[0]//2:glasses_image.shape[0] + min(left_eye[1],right_eye[1]) + glasses_image.shape[0]//2, left_eye[0] - med_width//2:glasses_image.shape[1] + left_eye[0] - med_width//2] = glasses_image
            mask = np.zeros((new_image.shape[0], new_image.shape[1]),dtype=np.uint8)
            mask[new_image[:,:,3] > 0] = 1
            new_image[mask < 1] = np.array([0,0,0,0])
            image[mask > 0] = np.array([0,0,0])
            new_image = new_image[:,:,:3]
            image = cv2.add(image, new_image)
    return image

def deal_with_it(image, face_classifier, eyes_classifier, glasses_image):
    image = image.copy()
    face_rects = detect_faces(image, face_classifier)
    eye_rects = detect_eyes(image, face_rects, eyes_classifier)
    image = draw_glasses(image, eye_rects, glasses_image)
    return image

cap = cv2.VideoCapture(0)

face_cascade = 'haarcascades/haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(face_cascade)

eyes_cascade = 'haarcascades/haarcascade_eye.xml'
eyes_classifier = cv2.CascadeClassifier(eyes_cascade)

glasses_image = cv2.imread("imgs/dealwithit.png", -1)
glasses_image = delete_borders(glasses_image)

while True:
    ret, frame = cap.read()
    frame = frame[:, ::-1]
    
    result_image = deal_with_it(frame, face_classifier, eyes_classifier, glasses_image)

    cv2.imshow("Result image", result_image)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()