# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, ZeroPadding2D)
from concise.metrics import tpr

from keras.applications.resnet50 import ResNet50
from keras import backend as K
from sklearn.metrics import fbeta_score, f1_score
from tensorflow.python.ops import math_ops

import tensorflow as tf
import functools
import numpy as np


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    #precision = precision(y_true, y_pred)
    #recall = recall(y_true, y_pred)
    #return 2*((precision*recall)/(precision+recall))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred , 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    r = true_positives / (possible_positives + K.epsilon())
    beta = 1 # fmeasure
    bb = beta**2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score



def f2(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    beta = 2.0
    beta2 = beta ** 2
    #result = ((1 + beta2) * math_ops.multiply(p, r) / (math_ops.multiply(beta2, p) + r + K.epsilon()))
    #K.set_value(p, 0)
    #return result
    #return ((1 + beta2) * math_ops.multiply(p, r) / (math_ops.multiply(beta2, p) + r + K.epsilon()))
    return ((1 + beta2) * p * r / (beta2 * p + r + K.epsilon()))


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def mpc_a(y_true, y_pred):
    return tf.metrics.mean_per_class_accuracy(y_true, y_pred, 2)


def make_binary_metric(metric_name, metric_func, num_classes, y_true, preds_one_hot):
    """Create a binary metric using `metric_func`"""
    overall_met = [None for _ in range(num_classes)]
    with tf.name_scope(metric_name):
        for cc in range(num_classes):
            #Metrics should take 1D arrays which are 1 for positive, 0 for negative
            two_true, two_pred = y_true[:, cc], preds_one_hot[:, cc]
            cur_met = metric_func(two_true, two_pred)
            tf.summary.scalar('%d' % cc, cur_met)
 
            return cur_met

            overall_met[cc] = cur_met
 
        #tf.summary.histogram('overall', overall_met)
    return K.sum(overall_met)/num_classes


def gg(y_true, y_pred):

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
 
    """Create precision, recall, and fmeasure metrics. Log them directly using tensorflow"""
    num_classes = K.get_variable_shape(y_pred)[1]
    mmm = K.get_variable_shape(y_pred)
    #print("+++++++++++++++++++++++++++ >>>>>>>>>>>> ", mmm)

    preds_cats = K.argmax(y_pred, axis=1)
    preds_one_hot = K.one_hot(preds_cats, num_classes)

    #r = make_binary_metric('precision', precision, num_classes, y_true, preds_one_hot)
    #make_binary_metric('recall', recall, num_classes, y_true, preds_one_hot)
    #make_binary_metric('fmeasure', fmeasure, num_classes, y_true, preds_one_hot)

    """Create a binary metric using `metric_func`"""
    metric_name = 'precision'
    metric_func = precision
    overall_met = [None for _ in range(num_classes)]
    with tf.name_scope(metric_name):
        for cc in range(num_classes):
            print("\n  cc = ", cc)
            #Metrics should take 1D arrays which are 1 for positive, 0 for negative
            two_true, two_pred = y_true[:, cc], preds_one_hot[:, cc] 
            cur_met = metric_func(two_true, two_pred)
            tf.summary.scalar('%d' % cc, cur_met)
 
            #return cur_met

            overall_met[cc] = cur_met
 
        #tf.summary.histogram('overall', overall_met)
    #print("\n  overall = ", overall_met)

    return overall_met[0] #cur_met




def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2 

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))


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
            #precision = as_keras_metric(tf.metrics.precision)
            #recall = as_keras_metric(tf.metrics.recall)
            #auc_roc = as_keras_metric(tf.metrics.auc)
            #model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, auc_roc, f1, f2_score])
            model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            #model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy', f2_score])
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
