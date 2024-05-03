import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('models/Autopilot_new.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1)[0, 0])
    steering_angle = steering_angle * 60
    return steering_angle


def keras_process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


steer = cv2.imread('resources/steering_wheel_image.png', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture('resources/run.mp4', cv2.CAP_AVFOUNDATION)
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret: 
        break
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    steering_angle = keras_predict(model, gray)
    # print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
    smoothed_angle = 0.9 * smoothed_angle + 0.1 * steering_angle

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    # print("Steering angle: ", steering_angle, "Smoothed angle: ", smoothed_angle)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
