from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import os
import fnmatch
import cv2
import random
import numpy as np
import string
import time
import json
import shutil
import base64
# import matplotlib.pyplot as plt

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*4))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
print(tf.__version__)


current_directory_path = pathlib.Path("./").absolute()
random_test_path = pathlib.Path(os.path.join(
    str(current_directory_path), 'data/test'))


class ModelPredict:
    def __init__(self):
        self.char_list = ' !"#%\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvxyzÀÁÂÊÔÚàáâãèéêìíòóôõùúýăĐđĩũƠơƯưạẢảẤấẦầẩẫậắằẳẵặẹẻẽếỀềỂểễỆệỉịọỏỐốồổỗộớờỞởỡợụỦủỨứừửữựỳỵỷỹ'
        self.random_img = []
        self.current_directory_path = pathlib.Path("./").absolute()
        self.random_test_path = pathlib.Path(
            os.path.join(str(current_directory_path), 'data/test'))
        self.loadActModel()

    def loadImage(self, uriImage):
        encoded_data = uriImage.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)

        img = cv2.imdecode(
            nparr, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('test.png', img) 

        
        return img

    def preprocessImg(self, img):
        
        height, width = img.shape
        img = cv2.resize(img, (1957, 273), interpolation=cv2.INTER_AREA)
        height, width = img.shape
        cv2.imwrite('resize.png', img) 
        # img = np.pad(img, ((0,0),(0, 1957-width)), 'median')

        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        img = np.expand_dims(img, axis=2)
        img = img/255.

      
        return img

    def predict(self, uriImage):
       img = self.loadImage(uriImage)
       img = self.preprocessImg(img)
       random_img = []
       random_img.append(img)
       random_img = np.array(random_img)
       prediction = self.act_model.predict(random_img)

       out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                      greedy=True)[0][0])
       # see the results
       all_predictions = []
       i = 0
       for x in out:
        # print("predicted text = ", end='')
        pred = ""
        for p in x:
            if int(p) != -1:
                pred += self.char_list[int(p)]
        all_predictions.append(pred)
        i += 1
       print(all_predictions)
       return all_predictions[0]
    
    def loadActModel(self):
        inputs = Input(shape=(273, 1957, 1))
        # convolution layer with kernel size (3,3)  #(273,1957,64)
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        # # poolig layer with kernel size (2,2) to make the height/2 and width/2  #(137,979,64)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        # #(137,979,128)
        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        # # poolig layer with kernel size (2,2) to make the height/2 and width/2  #(69,490,128)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        # #(69,490,256)
        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
        # # poolig layer with kernel size (2,2) to make the height/2  #(35,245,256)
        pool_3 = MaxPool2D(pool_size=(2, 2))(conv_3)
        batch_norm_3 = BatchNormalization()(pool_3)

        # #(35,245,256)
        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(batch_norm_3)
        # # poolig layer with kernel size (2,2) to make the height/2  #(18,123,256)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
        # # Batch normalization layer #3,9,256)
        batch_norm_5 = BatchNormalization()(pool_4)

        # (18, 123,512)
        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        # # poolig layer with kernel size (2,2) to make the height/2 #(9,62,512)
        pool_6 = MaxPool2D(pool_size=(2, 1))(conv_6)
        batch_norm_6 = BatchNormalization()(pool_6)

        # # poolig layer with kernel size (2,2) to make the height/2 #(5,31,512)
        pool_7 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
        batch_norm_7 = BatchNormalization()(pool_7)
        # # poolig layer with kernel size (2,2) to make the height/2 #(2,16,512)
        pool_8 = MaxPool2D(pool_size=(2, 1))(batch_norm_7)
        pool_9 = MaxPool2D(pool_size=(2, 1))(pool_8)

        # # # to remove the first dimension of one: (1, 8, 512) to (8, 512)
        squeezed = Lambda(lambda x: K.squeeze(x, 1))(pool_9)

        # # # # bidirectional LSTM layers with units=1957
        blstm_1 = Bidirectional(
            LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)

        # # # this is our softmax character proprobility with timesteps
        outputs = Dense(163, activation='softmax')(blstm_2)

        # # model to be used at test time

        # outputs = batch_norm_6
        act_model = Model(inputs, outputs)

         # define the length of input and label for ctc
        labels = Input(name='the_labels', shape=158, dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        
        # define a ctc lambda function to take arguments and return ctc_bach_cost
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            """
            labels: tensor (number of samples, max_string_length) containing the truth labels.
            y_pred: tensor (number of samples, time_steps, num_character_labels) containing the prediction, or output of the softmax.
            input_length: tensor (number of samples, 1) containing the sequence length for each batch item in y_pred.
            label_length: tensor (number of samples, 1) containing the sequence length for each batch item in y_true.
            """
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        
        # out loss function (just take the inputs and put it in our ctc_batch_cost)
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

        #model to be used at training time
        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

        act_model.load_weights('./model/model_new_512.h5')
        self.act_model = act_model

       
