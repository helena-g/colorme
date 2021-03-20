import os
import random

import keras
import numpy as np
import cv2
from keras_preprocessing.image import ImageDataGenerator
from sklearn.externals._pilutil import imsave
from sklearn.pipeline import Pipeline
import tensorflow as tf

from sklearn.svm import SVC
from tensorflow import int16
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet import preprocess_input

from data.model1 import Model1, Model2, Model3
from transformers.ColorHistTransformer import ColorHistTransformer

train_data_path = './data/color/train2'
test_data_path = './data/color'
data_path = './data/color'
dim = (256, 256)
def load_files(data_path):
    photos = []
    for file in  os.listdir(data_path):
        if '.jpg' not in file:
            continue
        x = cv2.imread(os.path.join(data_path,file))
        resized = cv2.resize(x, dim, interpolation=cv2.INTER_AREA)
        photos.append(resized)
    return photos


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_photos = np.array(load_files(train_data_path))[:4000]

    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        dtype='uint8'
    )

    #train_photos_lab = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2Lab) for image in  train_photos])
    # del train_photos
    #
    test_photos_color = np.array(load_files(test_data_path))
    test_photos_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in test_photos_color])
    test_photos_lab = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2Lab) for image in  test_photos_color])
    # del test_photos_color



    #
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, dim, [0, dim[0], 0, dim[1]])
    #
    # hist = cv.calcHist([img], [0], None, [256], [0, 256])


    # clf = SVC(random_state=9)
    # transformer = ColorHistTransformer()
    model = Model3()
    # Pipeline(
    #     [('hist_color_transformer', transformer),
    #      ('clf', clf),
    #      ])
    for i in [5]: #, 10, 100, 1000]:
        generated = test_photos_bw

        inception = InceptionResNetV2(weights='imagenet', include_top=True)
        #inception.graph = get_default_graph()
        #inception.load_weights('./data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')

        def create_inception_embedding(grayscaled_rgb):
            grayscaled_rgb_resized = []
            for i in grayscaled_rgb:
                i = cv2.resize(i, (299, 299), interpolation=cv2.INTER_AREA)
                #grayscaled_rgb_resized.append( np.dstack((i,i,i)))
                grayscaled_rgb_resized.append(cv2.cvtColor(i, cv2.COLOR_GRAY2RGB ))
                #cv2.imwrite("img_train_{}.png".format(0), grayscaled_rgb_resized)
            grayscaled_rgb_resized = preprocess_input(np.array(grayscaled_rgb_resized))
            embed= inception.predict(grayscaled_rgb_resized)
            return embed

        def image_a_b_gen(Xtrain, batch_size, datagen):
            #just remove embed in model1 or model2
            for batch in datagen.flow(Xtrain, batch_size=batch_size, save_to_dir='./data/tmp'):
                embed = create_inception_embedding( np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in batch]))
                lab_batch = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2Lab) for image in batch])
                X_batch = lab_batch[:, :, :, 0]
                X_batch = X_batch.reshape(X_batch.shape + (1,))
                Y_batch = lab_batch[:, :, :, 1:]
                Y_batch = (Y_batch.astype(int) - 128) / 128.0
                yield ([X_batch, embed], Y_batch)

        model.model.fit(image_a_b_gen(train_photos, 16, datagen),
                    steps_per_epoch=  train_photos.shape[0]/16 , epochs=i)
        # model.model.fit(datagen.flow(train_photos_lab[:,:, :, 0:1], train_photos_lab[:, :, :, 1:]/256, batch_size=16),
        #                 steps_per_epoch=  train_photos_lab.shape[0]/16 ,
        #                     epochs=i)
        #print(model.model.evaluate(X, Y, batch_size=1))  # Output colorizations
        color_me_embed = create_inception_embedding(generated)
        output = model.model.predict([generated[0:1, :, :, np.newaxis], color_me_embed[0:1,:]])
        output = output * 128 + 128
        canvas = np.zeros((256, 256, 3))
        canvas[:, :, 0] = generated[0][:, :]
        canvas[:, :, 1:] = output
        #cv2.imwrite("img_result2.png", canvas)
        cv2.imwrite("img_test_{}.png".format(i), generated[0])
        canvas = canvas.astype(int)
        #imsave("img_gray_scale.png", rgb2gray(lab2rgb(canvas)))
        cv2.imwrite("img_result{}.png".format(i), cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_Lab2RGB))

        #X = X.reshape(1, 400, 400, 1)
        #Y = Y.reshape(1, 400, 400, 2)
        #model.fit()

    keras.backend.clear_session()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
