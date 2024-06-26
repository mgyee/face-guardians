import tensorflow as tf
import matplotlib.pyplot as plt
import keras
# from keras.layers import Conv2D, Dropout, Flatten, Dense, LeakyReLU, MaxPool2D
# from keras.models import Sequential
# from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from preprocessing import preprocess_fer
import numpy as np

num_classes = 7 # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 1024
epochs = 10
checkpoint_callback = ModelCheckpoint(
    filepath='weights.h5',  # Specify the file path to save the weights
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,     # Save only the best model based on the monitored quantity
    save_weights_only=True,  # Save only the weights
    verbose=1)               # Show progress

def create_model() :
    model = keras.models.Sequential()
        
    model.add(keras.layers.Conv2D(64,(3,3), input_shape=(48,48, 1), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.20))
    
    model.add(keras.layers.Conv2D(128,(5,5), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.20))

    model.add(keras.layers.Conv2D(512,(3,3), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.20))
    
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))
    
    model.add(keras.layers.Conv2D(256,(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128,(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))
    
    #model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Dense(7,activation='softmax'))
    
    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=lr) , metrics=['accuracy'])
    return model

#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

def main():
    x_train, y_train, x_test, y_test = preprocess_fer()
    train_images, val_images, train_labels, val_labels = train_test_split(x_train, y_train, test_size=0.2)
    # gen = ImageDataGenerator()
    # train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    train_loader = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_loader = train_loader.batch(batch_size)

    model = create_model()

    #------------------------------

    # metric accuracy will tell us accuracy for each epoch
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    #------------------------------

    model.fit(train_loader.repeat(), steps_per_epoch=batch_size, epochs=epochs, 
              callbacks=[checkpoint_callback], validation_data=(val_images, val_labels))
    #overall evaluation
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', 100*score[1])

    monitor_testset_results = True

    if monitor_testset_results == True:
        #make predictions for test set
        predictions = model.predict(x_test)

        index = 0
        # very small range of predictions
        for i in predictions:
            if index < 30 and index >= 20:
                #print(i) #predicted scores
                #print(y_test[index]) #actual scores
                
                testing_img = np.array(x_test[index], 'float32')
                testing_img = testing_img.reshape([48, 48]);
                
                plt.gray()
                plt.imshow(testing_img)
                plt.show()
                
                print(i)
                
                emotion_analysis(i)
                print("----------------------------------------------")
            index = index + 1

    return model

if __name__ == "__main__":
    main()
