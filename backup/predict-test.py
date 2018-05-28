from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)

from keras.applications.vgg16 import VGG16

from keras.optimizers import SGD, RMSprop

import numpy as np
import cv2
import os

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from keras import backend as K

server = True
n_classes = 2 

images_filenames = 'images_filenames.txt'
labels = 'binary_classification.txt'

if server:
    # Server path
    test_img_path = '/home/master/tfm/ExperCNN/dataset_16_5/dataset_16_5-kfold1/test'
    dataset_directory = '/home/master/tfm/ExperCNN'
else:
    # Local machine path
    test_img_path = '/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/dataset_22_12/dataset_22_12-kfold1/test/'
    dataset_directory = '/home/willytell/Documentos/MCV/M9/TFM/ExperCNN'


# input of the VGG16
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
    img_shape = (3, img_width, img_height)
else:
    img_shape = (img_width, img_height, 3)    # 'channels_last'

#img_shape = (224, 224, 3)

def VGG_16(weights_path=None):
    #base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=img_shape)

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=img_shape)) #(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    # Add final layers
    #x = base_model.output
    #x = Flatten(name="flatten")(x)
    #x = Dense(4096, activation='relu', name='dense_1')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(4096, activation='relu', name='dense_2')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(n_classes, name='dense_3_{}'.format(n_classes))(x)
    #predictions = Activation("softmax", name="softmax")(x)

    #model = Model(input=base_model.input, output=predictions)

    if weights_path:
        model.load_weights(weights_path)

    return model


def read_data():
    X = np.genfromtxt(os.path.join(dataset_directory, images_filenames), dtype='str')
    y = np.genfromtxt(os.path.join(dataset_directory, labels), dtype='str')

    return X, y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confmatrix.jpg')
    

if __name__ == "__main__":

    # Read all the dataset
    X_all, y_all = read_data()

    # Read Neoplasicos test images directory
    neo_img_list = sorted(os.listdir(os.path.join(test_img_path, 'NEOPLASICO')))
    neo_img_arr = np.asarray(neo_img_list)
    print("Neoplasico lenght: ", len(neo_img_arr))

    # Read NOneoplasicos test images directory
    NOneo_img_list = sorted(os.listdir(os.path.join(test_img_path, 'NONEOPLASICO')))
    NOneo_img_arr = np.asarray(NOneo_img_list)
    print("NOneoplasico lenght: ", len(NOneo_img_arr))

    X_test = np.concatenate((neo_img_list, NOneo_img_arr), axis=0)

    y_test = np.array([])

    #np.save()

    for img_filename in X_test:
        idx = np.where(X_all == img_filename )
        idx_arr = np.asarray(idx[0])  # tupla to np.array
        classification = y_all[idx_arr]
        if classification == 'NEOPLASICO':
            y_test = np.append(y_test, 1)
        else:
            y_test = np.append(y_test, 0)

    print("y_test = ", y_test)
    print("len(y_test) = ", len(y_test))

    # Test pretrained model
    model = VGG_16('weights.hdf5')
    rmsprop = RMSprop(lr=0.00001, rho=0.9, decay=0.0)
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

    # Dimensions for test_sample
    image = load_img(os.path.join(test_img_path, 'NEOPLASICO', neo_img_arr[0]), target_size=(224, 224))
    image = img_to_array(image)
    test_samples = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #print("len(test_samples) = ", len(test_samples))
    test_samples = np.delete(test_samples, 0, axis=0)
    #print("len(test_samples) = ", len(test_samples))

    # Predict Neoplasico images
    for img_filename in neo_img_arr:
        image = load_img(os.path.join(test_img_path, 'NEOPLASICO', img_filename), target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        test_samples = np.concatenate((test_samples, image), axis=0)


    # Predict NOneoplasico images
    for img_filename in NOneo_img_arr:
        image = load_img(os.path.join(test_img_path, 'NONEOPLASICO', img_filename), target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        test_samples = np.concatenate((test_samples, image), axis=0)


    predictions = model.predict(test_samples, batch_size=10, verbose=0)
    print ("np.argmax(predicitons) ", np.argmax(predictions))
    print ("predictions = ", predictions)
    rounded_pred_model = np.array([], dtype=np.int64)
    for p in predictions:
        rounded_pred_model=np.append(rounded_pred_model, np.argmax(p))

    print ("rounded_pred_model", rounded_pred_model)


    rounded_predictions = model.predict_classes(test_samples, batch_size=10, verbose=0)
    print ("rounded_predictions", rounded_predictions)

    cm = confusion_matrix(y_test, rounded_predictions)
    cm_plot_labels = ['Noneoplasico','Neoplasico']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')



