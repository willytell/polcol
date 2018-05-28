from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)

from keras.applications.vgg16 import VGG16

from keras.optimizers import SGD, RMSprop

import numpy as np
import cv2

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


n_classes = 2
img_shape = (224, 224, 3)  # (224, 244 <<----- HERE WAS THE ERROR, 3)

def VGG_16(weights_path=None):
    base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='dense_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='dense_3_{}'.format(n_classes))(x)
    predictions = Activation("softmax", name="softmax")(x)

    model = Model(input=base_model.input, output=predictions)

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('CLINIC-CVC0032_0003_30_07.bmp'), (224, 224)).astype(np.float32)
    # im[:, :, 0] -= 103.939
    # im[:, :, 1] -= 116.779
    # im[:, :, 2] -= 123.68
    # im = im.transpose((1, 0, 2))
    # im = np.expand_dims(im, axis=0)

    image = load_img('CLINIC-CVC0032_0003_30_07.bmp', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)

    im = image


    # Test pretrained model
    model = VGG_16('weights.hdf5')
    rmsprop = RMSprop(lr=0.00001, decay=0.0)
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    out = model.predict(im)
    print ("output: ", np.argmax(out))

