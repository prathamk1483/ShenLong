from mss import mss
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import Model
import os
from keras.models import load_model
from directkeys import X,Z,A,D,PressKey , ReleaseKey
import time


def ZZZ():
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    



def XXZZ():
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    


def XXXX():
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)
    PressKey(X)
    time.sleep(0.05)
    ReleaseKey(X)
    time.sleep(0.05)

def RRZZ():
    PressKey(D)
    time.sleep(0.1)
    ReleaseKey(D)
    time.sleep(0.1)
    PressKey(D)
    time.sleep(0.1)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05)
    ReleaseKey(D)
    time.sleep(0.1)
    
def RZ():
    PressKey(D)
    time.sleep(0.1)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    time.sleep(0.05) 
    ReleaseKey(D)
    time.sleep(0.1)

def LZ():
    PressKey(A)
    time.sleep(0.1)
    PressKey(Z)
    time.sleep(0.05)
    ReleaseKey(Z)
    ReleaseKey(A)
    time.sleep(0.1)


# Define the resolution for resizing
target_width = 256
target_height = 144

# Bounding box for capturing the screen
bbox = {'top': 100, 'left': 0, 'width': 700, 'height': 480}
sct = mss()

output_dir = "D:/Bloody Roar 2 project/Dataset/images"
frame_count = 0
model = YOLO('yolov8n-pose.pt')
model2 = load_model("6out.h5")


while True:
    # Capture the screen
    image = np.array(sct.grab(bbox))

    # Apply Gaussian Blur to hide unneccesary details
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.GaussianBlur(image,(5,5),0)

    if image.shape[2] == 4:  
        image = image[:, :, :3]

    results = model(source=image, show=False, conf=0.25)
    result = results[0]
    annotated_image = result.plot()
    cv2.imshow("BONES",annotated_image)
    # output_path = os.path.join(output_dir, f"annotated_frame_{frame_count}.jpg")
    # cv2.imwrite(output_path,annotated_image)
    # print(annotated_image.shape)

    annotated_image = np.expand_dims(annotated_image, axis=0)
    
    r = model2.predict(annotated_image)
    print(r)
    r = np.argmax(r[0])
    print(r)
    if r== 0:
        ZZZ()
    elif r == 1:
        XXXX()
    elif r == 2:
        XXZZ()
    elif r == 3:
        RRZZ()
    elif r == 4:
        RZ()
    elif r == 5:
        LZ()
    else:
        print("Unknown")
    frame_count += 1

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
