# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras import backend as K


# Paper: https://arxiv.org/pdf/1409.1556.pdf

class VGG(object):
    def __init__(self, num_classes, optimizer):
        self.num_classes = num_classes
        self.optimizer = optimizer

    def build_vgg(self, cf, img_rows=None, img_cols=None, input_channels=3, n_layers=16, load_pretrained=False,
                  freeze_layers_from='base_model'):

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
        if n_layers == 16:
            base_model = VGG16(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)
        elif n_layers == 19:
            base_model = VGG19(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)
        else:
            raise ValueError('Number of layers should be 16 or 19')

        # Add final layers
        x = base_model.output
        x = Flatten(name="flatten")(x)
        x = Dense(4096, activation='relu', name='dense_1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='dense_2')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, name='dense_3_{}'.format(self.num_classes))(x)
        predictions = Activation("softmax", name="softmax")(x)

        # This is the model we will train
        model = Model(input=base_model.input, output=predictions, name="VGG" + str(n_layers))

        # Compile model
        # For a binary classification problem
        if self.num_classes == 2:
            model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
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
