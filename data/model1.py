import cv2
from cv2.cv2 import resize
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.engine.input_layer import InputLayer, Input
from tensorflow.python.keras.layers import Conv2D, UpSampling2D, RepeatVector, Reshape, concatenate, Dropout
from tensorflow.python.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
from keras.applications.inception_resnet_v2 import preprocess_input

class Model1:

    def __init__(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model

class Model2:

    def __init__(self):
        # Design the neural network
        model = Sequential()
        model.add(InputLayer(input_shape=(256, 256, 1)))
        #model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        #model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))  # Finish model
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model

class Model3:
    # Encoder

    def __init__(self):
        #self.inception = InceptionResNetV2(weights=None, include_top=True)
        #self.inception.load_weights('/data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
        #inception.graph = tf.get_default_graph()

        embed_input = Input(shape=(1000,))
        encoder_input = Input(shape=(256, 256, 1,))
        encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)  # Fusion

        fusion_output = RepeatVector(32 * 32)(embed_input)
        fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
        fusion_output = concatenate([encoder_output, fusion_output], axis=3)
        fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)  # Decoder
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)

        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)

        model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model


