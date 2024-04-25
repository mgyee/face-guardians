import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D, Dropout, Flatten, Dense, LeakyReLU, MaxPool2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from preprocessing import preprocess_fer
import numpy as np

num_classes = 7 # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 1024
epochs = 100
checkpoint_callback = ModelCheckpoint(
    filepath='weights.h5',  # Specify the file path to save the weights
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,     # Save only the best model based on the monitored quantity
    save_weights_only=True,  # Save only the weights
    verbose=1)               # Show progress

def create_model() :
    model = Sequential([
        Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
        Conv2D(64, 3, 1, padding="same",activation="relu", name="block1_conv2"),
        MaxPool2D(2, name="block1_pool"),

        # Block 2
        Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
        Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
        MaxPool2D(2, name="block2_pool"),

        # Block 3
        Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
        Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
        Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
        MaxPool2D(2, name="block3_pool"),

        # Block 4
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
        MaxPool2D(2, name="block4_pool"),

        # Block 5
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
        Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
        MaxPool2D(2, name="block5_pool"),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.4),
        Dense(512, activation="relu"),
        Dropout(0.4),
        Dense(15, activation="softmax")
    ])

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
    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    model = create_model()

    #------------------------------

    # metric accuracy will tell us accuracy for each epoch
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    #------------------------------

    model.fit(train_generator, steps_per_epoch=batch_size, epochs=epochs, callbacks=[checkpoint_callback])
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

# def temp():
    # okay model:
    #1st convolution layer
    # model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    # model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    # #2nd convolution layer
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    # #3rd convolution layer
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    # model.add(Flatten())

    # #fully connected neural networks
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))

    # model.add(Dense(num_classes, activation='softmax'))
    pass
    # really bad model
    # model = Sequential([
    #     Conv2D(64, 3, 1, padding="same"),
    #     LeakyReLU(alpha=0.2),
    #     BatchNormalization(),

    #     Conv2D(64, 3, 1, padding="same"),
    #     LeakyReLU(alpha=0.2),
    #     BatchNormalization(),
    #     MaxPooling2D((2, 2)),
    #     Dropout(0.25),

    #     Conv2D(32, 3, 1, padding="same"),
    #     LeakyReLU(alpha=0.2),
    #     BatchNormalization(),

    #     Conv2D(32, 3, 1, padding="same"),
    #     LeakyReLU(alpha=0.2),
    #     BatchNormalization(),

    #     Conv2D(32, 3, 1, padding="same"),
    #     LeakyReLU(alpha=0.2),
    #     BatchNormalization(),
    #     MaxPooling2D((2, 2)),
    #     Dropout(0.25),

    #     # flatten to start the classification
    #     Flatten(),
    #     Dense(2048),
    #     LeakyReLU(alpha=0.2),

    #     Dense(1024),
    #     LeakyReLU(alpha=0.2),
    
    #     Dropout(0.25),
    #     Dense(15, activation="softmax")
    # ])