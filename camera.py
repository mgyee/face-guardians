import time
from misc.original_model import FacialExpressionModel
from utils import overlay_image
import numpy as np
import random
import cv2
import sys

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('misc/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_pairs = {"angry" :"data/images/angry.png", "disgust": "data/images/disgust.png",
                 "fear": "data/images/fear.png", "happy": "data/images/happy.png",
                 "sad": "data/images/sad.png", "surprise": "data/images/surprise.png",
                 "neutral": "data/images/neutral.png"}

checkmark_file = "data/images/checkmark.png"
end_screen_file = "data/images/lock.png"

rect_width = 200
rect_height = 50
rect_position = (0, 0, rect_width, rect_height)
sleep_secs = 2

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """

    # returns bool (read successfully or not), frame data
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # detect the faces, (image, scaleFactor, minNeighbors), tune scale factor,
    # smaller scale factor, greater chance of detecting faces, more computational
    # minNeighbors, higher value results in less detections but with higher quality
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def get_next_emotion(pwd_list):
    emotion = pwd_list.pop(0)
    return (emotion, emotion_pairs[emotion])

def time_stall(emotion, emoji, checkmark, start_time, seconds):
    """
    Stall for a certain number of seconds.

    Args:
        start_time: The time at which the stall started.
        seconds: The number of seconds to stall for.
    """

    while (time.time() < start_time + seconds):
        _, fr = rgb.read()
        fr_width = fr.shape[1]
        cv2.rectangle(fr, (fr_width - rect_width, 0), (fr_width, rect_height), (255, 255, 255), -1)
        cv2.putText(fr, emotion, (fr_width - rect_width + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        fr = overlay_image(fr, emoji, position=(0, 0))
        fr = overlay_image(fr, checkmark, position=(0, 128))
        cv2.imshow('Filter', fr)
        cv2.waitKey(2)

def time_stall_end(end_screen, start_time, seconds, position=(0, 128)):
    while (time.time() < start_time + seconds):
        _, fr = rgb.read()
        fr = overlay_image(fr, end_screen, position)
        cv2.imshow('Filter', fr)
        cv2.waitKey(2)

def start_app(cnn):
    checkmark = cv2.imread(checkmark_file, -1)
    end_screen = cv2.imread(end_screen_file, 1)
    detected_emotion = None
    start_time = None
    skip_frame = 10
    data = []
    flag = False
    ix = 0

    while True:
        user_input = input("Set Password (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral): ").split(",")
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        password = []
        flag = False

        for emotion in user_input:
            emotion = emotion.strip().lower()
            if emotion not in emotions:
                print("Please enter valid emotion(s): Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")
                flag = True
                break
            else:   
                password.append(emotion)
        if flag:
            continue
        break
    
    next_emotion, emoji_file = get_next_emotion(password)

    while True:
        ix += 1
        faces, fr, gray_fr = __get_data__()
        fr_width = fr.shape[1]

        # center the text inside the rectangle
        # putText(image, text, (x val, y val), font type, font size, color, thickness)
        # negative thickness is a fill
        cv2.rectangle(fr, (fr_width - rect_width, 0), (fr_width, rect_height), (255, 255, 255), -1)

        # cv2.FONT_HERSHEY_SIMPLEX seems best, others look too pixelated
        cv2.putText(fr, next_emotion, (fr_width - rect_width + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        emoji = cv2.imread(emoji_file, -1)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            # region of interest
            roi = cv2.resize(fc, (48, 48))
            detected_emotion = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis]).lower()

            cv2.putText(fr, detected_emotion, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break

        fr = overlay_image(fr, emoji, position=(0, 0))
        cv2.imshow('Filter', fr)

        if (detected_emotion == next_emotion):
            start_time = time.time()
            fr = overlay_image(fr, checkmark, position=(0, 128))
            cv2.imshow('Filter', fr)

            time_stall(next_emotion, emoji, checkmark, start_time, sleep_secs)
            
            if password:
                next_emotion, emoji_file = get_next_emotion(password)
            else:
                fr = overlay_image(fr, end_screen, position=(0, 128))
                cv2.imshow('Filter', fr)
                start_time = time.time()
                time_stall_end(end_screen, start_time, sleep_secs, position=(0, 128))
                
                break



    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "weights.h5")
    start_app(model)