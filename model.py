from pathlib import Path
from statistics import mode
import numpy as np
import pandas as pd
import cv2
from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpers.audio_encoder import AudioFeatureExtraction

class Model():

    def __init__(self):
        self.audioFeatureExtractor = AudioFeatureExtraction(30)
        self.audio_path = "data/audio/"
        self.extracted_face_path = "data/extracted_face/"
        self.dimension = (100, 100, 3)
        self.n_dim = 132300

    def create_dataset(self):
        x_train = []
        y_train = []
        audioNameVsAudio = {}
        for file in os.listdir(self.audio_path):
            audioNameVsAudio[Path(file).resolve().stem] = self.audioFeatureExtractor.feature_extraction(file)
        for file in os.listdir(self.extracted_face_path):
            audio_fileName = Path(file).resolve().stem[:-1]
            if audio_fileName in audioNameVsAudio.keys():
                x_train.append(audioNameVsAudio.get(audio_fileName))
                y_train.append(cv2.imread(self.extracted_face_path + file))
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train) / 255

    def create_generater_model(self):
        model = Sequential()
        model.add(Dense(625 , input_dim = self.n_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((25, 25, 1)))
        model.add(Conv2DTranspose(8, (3, 3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(8, (3, 3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (100, 100), activation='sigmoid', padding='same'))
        model.summary()
        self.generator_model = model

    def create_discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=self.dimension))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        self.disciminator_model = model
    
    def create_combined_model(self):
        model = Sequential()
        self.disciminator_model.trainable = False
        model.add(self.generator_model)
        model.add(self.disciminator_model)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        self.combined_model = model

    def train_generator_model(self):
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.generator_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(self.x_train.shape)
        print(self.y_train.shape)
        self.generator_model.fit(self.x_train, self.y_train, batch_size=8, epochs = 10, verbose = True)

    def save_generator_model(self):
        self.generator_model.save_weights("savedmodel/generator_model.h5")

    def load_generator_model(self):
        self.generator_model.load_weights("savedmodel/generator_model.h5")

    def test_generator_model(self): 
        input = self.audioFeatureExtractor.feature_extraction_test("test.wav", "testdata/")
        output = self.generator_model.predict(np.array([input]))
        print(output[0])
        print(self.y_train[0])
        cv2.imwrite("testdata/test.png", output[0] * 255)
    
    def train_combined_model(self, epochs = 1, batches = 4):
        for epoch in range(epochs):
            index = 0
            print(">>>> Current epoch : ", epoch + 1)
            currentBatch = 1
            while index <= len(self.y_train):
                print(">>>> Current batch: ", currentBatch)

                startIndex = index
                index += batches
                endIndex = min(index, len(self.y_train))
                
                x_discriminator, y_discriminator = self.generate_discriminator_training_set(10)
                print(">>>> Training in progress <<<<")
                d_loss, d_acc = self.disciminator_model.train_on_batch(x_discriminator, y_discriminator)
                
                y_combined_model = np.ones((endIndex - startIndex, 1))
                g_loss, g_acc = self.combined_model.train_on_batch(self.x_train[startIndex : endIndex], y_combined_model)

                print('>>>> %d, d[%.3f,%.3f], g[%.3f,%.3f]' % (currentBatch, d_loss, d_acc, g_loss ,g_acc))
                currentBatch += 1


    def generate_discriminator_training_set(self, batches = 4):
        x_train = []
        y_train = []
        for z in range(batches//2) :
            x_train.append(self.y_train[np.random.randint(0, len(self.y_train) - 1)])
            y_train.append(1)
            x_train.append(np.random.random((100, 100, 3)))
            y_train.append(0)
        return np.array(x_train), np.array(y_train)



if __name__ == "__main__":
    model = Model()
    model.create_dataset()
    model.create_generater_model()
    model.create_discriminator_model()
    # model.create_combined_model()
    # model.train_generator_model()
    model.create_combined_model()
    model.train_combined_model()
    model.save_generator_model()
    model.load_generator_model()
    model.test_generator_model()

    