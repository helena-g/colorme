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
from tensorflow.python.keras.layers import GaussianNoise
from tensorflow.python.ops.image_ops_impl import random_hue

from data.model1 import Model1, Model2, Model3
from transformers.ColorHistTransformer import ColorHistTransformer

train_data_path = './data/color/train_small'
test_data_path = './data/color/small'
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
        #cv2.imwrite("img_test_{}.png".format(file), resized)
    return photos


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# def _noisify(
#     x, pct_pixels_min: float = 0.001, pct_pixels_max: float = 0.4, noise_range: int = 30
# ):
#     if noise_range > 255 or noise_range < 0:
#         raise Exception("noise_range must be between 0 and 255, inclusively.")
#
#     h, w = x.shape[1:]
#     img_size = h * w
#     mult = 10000.0
#     pct_pixels = (
#         random.randrange(int(pct_pixels_min * mult), int(pct_pixels_max * mult)) / mult
#     )
#     noise_count = int(img_size * pct_pixels)
#
#     for ii in range(noise_count):
#         yy = random.randrange(h)
#         xx = random.randrange(w)
#         noise = random.randrange(-noise_range, noise_range)
#         x[yy, xx, 0].+=noise
#
#     return x

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_photos = np.array(load_files(train_data_path))
    sample = GaussianNoise(20, dtype=tf.float64)

    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=[0.75, 1],
        rotation_range=5,
        horizontal_flip=True,
        dtype='uint8',
        preprocessing_function=lambda x : random_hue(x, seed=9, max_delta=0.05)
    )
    # datagen = ImageDataGenerator(
    #     shear_range=0.0,
    #     zoom_range=0.0,
    #     rotation_range=0,
    #     horizontal_flip=False,
    #     dtype='uint8'
    # )

    #train_photos_lab = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2Lab) for image in  train_photos])
    # del train_photos
    #
    test_photos_color = np.array(load_files(test_data_path))
    test_photos_bw = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_photos_color])
    test_photos_lab = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2Lab) for image in  test_photos_color])
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
    for i in [30000]: #, 10, 100, 1000]:
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
            for batch in datagen.flow(Xtrain, batch_size=batch_size):#),  save_to_dir='./data/tmp'):
                grey_batch =  np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in batch])
                embed = create_inception_embedding(grey_batch)
                lab_batch = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2Lab) for image in batch])
                X_batch = grey_batch
                X_batch = X_batch.reshape(X_batch.shape + (1,))#.astype(float)
                Y_batch = lab_batch[:, :, :, 1:]
                Y_batch = (Y_batch.astype(int) - 128) / 128.0
                #yield (X_batch.astype(float), Y_batch.astype(float))
                yield ([X_batch, embed.astype(int)], Y_batch)

        #print("AAAAAAAAAAAAAAAAAA",train_photos.shape[0]/1)
        model.model.fit(image_a_b_gen(train_photos, 8, datagen),
                steps_per_epoch=  train_photos.shape[0]/8 , epochs=i)
        # model.model.fit(image_a_b_gen(train_photos, 1, datagen),
        #             steps_per_epoch=2 , epochs=i)

        #print(model.model.evaluate(X, Y, batch_size=1))  # Output colorizations
        color_me_embed = create_inception_embedding(generated)
        #output = model.model.predict([generated[0:1, :, :, np.newaxis].astype(float)])

        output = model.model.predict([generated[0:1, :, :, np.newaxis], color_me_embed[0:1,:].astype(int)])
        output = output * 128 + 128
        canvas = np.zeros((256, 256, 3))
        canvas[:, :, 0] = generated[0][:, :]
        canvas[:, :, 1:] = output
        #cv2.imwrite("img_result2.png", canvas)
        cv2.imwrite("img_test_{}.png".format(i), generated[0])
        canvas = canvas.astype(int)
        #imsave("img_gray_scale.png", rgb2gray(lab2rgb(canvas)))
        cv2.imwrite("img_result{}.png".format(i), cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_Lab2BGR))

        #X = X.reshape(1, 400, 400, 1)
        #Y = Y.reshape(1, 400, 400, 2)
        #model.fit()

        output = model.model.predict([generated[1:2, :, :, np.newaxis], color_me_embed[1:2,:].astype(int)])
        output = output * 128 + 128
        canvas = np.zeros((256, 256, 3))
        canvas[:, :, 0] = generated[1][:, :]
        canvas[:, :, 1:] = output
        #cv2.imwrite("img_result2.png", canvas)
        canvas = canvas.astype(int)
        #imsave("img_gray_scale.png", rgb2gray(lab2rgb(canvas)))
        cv2.imwrite("img_result_other{}.png".format(i), cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_Lab2BGR))


        output = model.model.predict([generated[2:3, :, :, np.newaxis], color_me_embed[2:3,:].astype(int)])
        output = output * 128 + 128
        canvas = np.zeros((256, 256, 3))
        canvas[:, :, 0] = generated[2][:, :]
        canvas[:, :, 1:] = output
        #cv2.imwrite("img_result2.png", canvas)
        canvas = canvas.astype(int)
        #imsave("img_gray_scale.png", rgb2gray(lab2rgb(canvas)))
        cv2.imwrite("img_result_other_other{}.png".format(i), cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_Lab2BGR))
    keras.backend.clear_session()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
