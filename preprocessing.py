import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

num_classes = 7 # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 1024
epochs = 100

def preprocess_fer() :
    #------------------------------
    #cpu - gpu configuration
    # needed? some of this is deprecated
    # config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
    # sess = tf.Session(config=config) 
    # keras.backend.set_session(sess)
    #------------------------------
    with open("data/fer2013.csv") as f:
    # with open("/home/mgyee/Downloads/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    #------------------------------
    #initialize trainset and test set
    x_train, y_train, x_test, y_test = [], [], [], []

    #------------------------------
    #transfer train and test set data
    for i in range(1,num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")
            
            val = img.split(" ")
                
            pixels = np.array(val, 'float32')
            
            emotion = keras.utils.to_categorical(emotion, num_classes)
        
            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("",end="")

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape, 'train samples shape')
    print(y_train.shape, 'train labels shape')
    print(x_test.shape, 'test samples shape')
    print(y_test.shape, 'test labels shape')

    return x_train, y_train, x_test, y_test
