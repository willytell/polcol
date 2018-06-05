# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)

from keras.applications.resnet50 import ResNet50
from keras import backend as K

import tensorflow as tf
import functools

def as_keras_metric(method):
    #import functools
    #from keras import backend as K
    #import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


# Paper: https://arxiv.org/pdf/1409.1556.pdf

class myResNet50(object):
    def __init__(self, num_classes, optimizer):
        self.num_classes = num_classes
        self.optimizer = optimizer

    def build_resnet50(self, cf, img_rows=None, img_cols=None, input_channels=3, load_pretrained=False, freeze_layers_from='base_model'):

        if K.image_dim_ordering() == 'th':
            img_shape = (input_channels, img_rows, img_cols)
            axis = 1
        else:
            img_shape = (img_rows, img_cols, input_channels)
            axis = 3

        # Decide if load pretrained weights from imagenet
        if load_pretrained:
            weights = 'imagenet'
        else:
            weights = None


        # Get base model
        base_model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)

        # Add final layers
        x = base_model.output
        x = Flatten(name="flatten")(x)
        x = Dense(self.num_classes, name='dense_{}'.format(self.num_classes))(x)
        predictions = Activation("softmax", name="softmax")(x)

        # This is the model we will train
        model = Model(input=base_model.input, output=predictions, name="ResNet50")

        # Compile model
        # For a binary classification problem
        if self.num_classes == 2:
            precision = as_keras_metric(tf.metrics.precision)
            recall = as_keras_metric(tf.metrics.recall)
            auc_roc = as_keras_metric(tf.metrics.auc)
            model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, auc_roc])
        else:
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Load pretrained weights
        if cf.load_pretrained:
            print('   loading model weights from: ' + cf.weights_file + '...')
            model.load_weights(cf.weights_file, by_name=True)

        # Show model structure
        if cf.show_model:
            model.summary()
            # plot_model(model, to_file=os.path.join(cf.savepath, 'model.png'))

            # Output the model
        print('   Model: ' + cf.model_name)

        return model
