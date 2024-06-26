from keras.models import model_from_json
import numpy as np
from model import create_model

# TODO: Rename this to model and rename current model.py into something else
# Probably something about creating and traing the model util/train?
class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        # with open(model_json_file, "r") as json_file:
            # loaded_model_json = json_file.read()
            # self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model = create_model()

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


if __name__ == '__main__':
    pass