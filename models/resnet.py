# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)
#from concise.metrics import tpr

from keras.applications.resnet50 import ResNet50
from keras import backend as K

import tensorflow as tf

 

# Paper: https://arxiv.org/pdf/1409.1556.pdf

class myResNet50(object):
    def __init__(self, num_classes, optimizer):
        self.num_classes = num_classes
        self.optimizer = optimizer

    def build_resnet50(self, cf, img_rows=None, img_cols=None, input_channels=3, load_pretrained=False, freeze_layers_from='base_model'):

        print('   Model: ' + cf.model_name)

        if K.image_dim_ordering() == 'th':
            img_shape = (input_channels, img_rows, img_cols)
            axis = 1
        else:
            img_shape = (img_rows, img_cols, input_channels)
            axis = 3

        # Decide if load pretrained weights from imagenet
        if load_pretrained:
            weights = 'imagenet'
            print('   Loading pretrained weights: imageNet')
        else:
            weights = None


        # Get base model
        base_model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)

        # Add final layers
        x = base_model.output
        x = Flatten(name="flatten")(x)

        #x = Dense(512, activation='relu',name='fc-1')(x)
        #x = Dropout(0.5)(x)
        #x = Dense(256, activation='relu',name='fc-2')(x)
        #x = Dropout(0.5)(x)
        #x = Dense(128, activation='relu',name='fc-3')(x)
        #x = Dropout(0.5)(x)

        x = Dense(self.num_classes, name='dense_{}'.format(self.num_classes))(x)
        predictions = Activation("softmax", name="softmax")(x)

        # This is the model we will train
        model = Model(input=base_model.input, output=predictions, name="ResNet50")

        # Freeze some layers
        if freeze_layers_from is not None:
            if freeze_layers_from == 'base_model':
                print ('   Freezing base model layers')
                for layer in base_model.layers:
                    layer.trainable = False
            else:
                for i, layer in enumerate(model.layers):
                    print(i, layer.name)
                print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
                for layer in model.layers[:freeze_layers_from]:
                   layer.trainable = False
                for layer in model.layers[freeze_layers_from:]:
                   layer.trainable = True


        # Compile model
        # For a binary classification problem
        if self.num_classes == 2:
            model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


        # Show model structure
        if cf.show_model:
            model.summary()
            # plot_model(model, to_file=os.path.join(cf.savepath, 'model.png'))


        return model
