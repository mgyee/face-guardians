from misc.original_model import FacialExpressionModel
from utils import overlay_image
import numpy as np
import random
import cv2

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('misc/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_pairs = [("Angry", "data/images/angry.png"), ("Disgust", "data/images/disgust.png"),
                 ("Fear", "data/images/fear.png"), ("Happy", "data/images/happy.png"),
                 ("Sad", "data/images/sad.png"), ("Surprise", "data/images/surprise.png"),
                 ("Neutral", "data/images/neutral.png")]

rect_width = 200
rect_height = 50
rect_position = (0, 0, rect_width, rect_height)

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

def start_app(cnn):
    next_emotion = random.choice(emotion_pairs)
    detected_emotion = None
    skip_frame = 10
    data = []
    flag = False
    ix = 0

    while True:
        ix += 1
        faces, fr, gray_fr = __get_data__()
        fr_width = fr.shape[1]

        # center the text inside the rectangle
        # putText(image, text, (x val, y val), font type, font size, color, thickness)
        # negative thickness is a fill
        cv2.rectangle(fr, (fr_width - rect_width, 0), (fr_width, rect_height), (255, 255, 255), -1)

        # cv2.FONT_HERSHEY_SIMPLEX seems best, others look too pixelated
        cv2.putText(fr, next_emotion[0], (fr_width - rect_width + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        emoji = cv2.imread(next_emotion[1], -1)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            # region of interest
            roi = cv2.resize(fc, (48, 48))
            detected_emotion = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, detected_emotion, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break

        fr = overlay_image(fr, emoji, position=(100, 100))
        cv2.imshow('Filter', fr)

        if (detected_emotion == next_emotion[0]):
            next_emotion = random.choice(emotion_pairs)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "weights.h5")
    start_app(model)