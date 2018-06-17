import os
import numpy as np
import random

from sklearn.metrics import recall_score, precision_score, fbeta_score, f1_score, cohen_kappa_score, average_precision_score, precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.utils.multiclass import unique_labels    

from sklearn.model_selection import StratifiedKFold
from scipy.misc import imread
import skimage.transform

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_data(dataset_directory, images_filenames, labels):
    X = np.genfromtxt(os.path.join(dataset_directory, images_filenames), dtype='str')
    y = np.genfromtxt(os.path.join(dataset_directory, labels), dtype='str')

    return X, y


def rate(total_elements, freq):
    assert total_elements != 0
    assert total_elements == np.sum(freq)

    proportion = np.copy(freq)

    proportion = np.divide(proportion, float(total_elements)) # FIXED: np.divide, scalar must be float!

    return proportion

def filter_by_class(X, y, filter):

    idx = np.where(y == filter)  # filter = 'NONEOPLASICO' | 'NEOPLASICO'
    idx_arr = np.asarray(idx[0])    # tupla to np.array

    return np.copy(X[idx_arr]), np.copy(y[idx_arr])

def shuffle (X, y):
    r = random.random()
    #print (r)
    random.shuffle(X, lambda: r)  # shuffle images_filenames
    random.shuffle(y, lambda: r)  # shuffle labels


def dataset_bins_idx(X, y, n_splits):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    #print("n_splits : ", skf.get_n_splits(X, y))
    #print(skf)

    for train_index, validation_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        yield validation_index[-1]



# Load image
def load_img(path, resize=None, order=1, preserve_range=True):
    # Load image
    img = imread(path)

    # Resize
    # print('Desired resize: ' + str(resize))
    if resize is not None:
        img = skimage.transform.resize(img, resize, order=order, preserve_range=preserve_range)
        print('Final resize: ' + str(img.shape))

    # Return image
    return img


# Precompute the mean and std
def compute_mean_std(directory, file_names, resize=None, rescale=0, method='mean', mean=None):
    sum, n = 0, 0
    # Process each file
    for file_name in file_names:
        # Load image and reshape as a vector
        #x = imread(os.path.join(directory, file_name))
        x = load_img(os.path.join(directory, file_name), resize=resize)

        if rescale:
            x = x * rescale
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        n += x.shape[0]

        # Compute mean or std
        if method == 'mean':
            sum += np.sum(x, axis=0)
        elif method == 'var':
            x -= mean
            sum += np.sum(x * x, axis=0)

    return sum/n

def plot_confusion_matrix(cm, classes, fname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized " + title)
    else:
        print(title + ', without normalization')

    print(cm)



    if fname is not None:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(fname) #('confmatrix.jpg')


def print_stats(y_true, rounded_pred, epoch=None, fname=None, normalize=None):

    labels = unique_labels(y_true, rounded_pred)
    sample_weight = None
    
    # ensure, that y_true has at least one 1, because sklearn's fbeta can't handle all-zeros
    #y_true[:, 0] += 1 - y_true.sum(axis=1).clip(0, 1)
    
    p, r, f1, s = precision_recall_fscore_support(y_true, rounded_pred, labels=labels, average=None, sample_weight=sample_weight)
    
    # f2-score
    beta=2
    
    f2_class0 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=0, average='binary', sample_weight=None)
    f2_class1 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=1, average='binary', sample_weight=None)
    
    # f2 score averaged
    f2 = np.average(np.array([f2_class0, f2_class1]), weights=s)
    print("\n")
    if epoch is not None: print("In epoch: {}".format(epoch+1))
    print("f2-score f2_class0: {:.6f}".format(f2_class0))
    print("f2-score f2_class1: {:.6f}".format(f2_class1))
    print("f2-score: {:.6f} ".format(f2))
    
    #return the fraction of correctly classified samples
    acc_norm = accuracy_score(y_true, rounded_pred, normalize=True, sample_weight=None)
    print("Normalized acc: {:.6f}".format(acc_norm))
    
    # return the number of correctly classified sample
    acc = accuracy_score(y_true, rounded_pred, normalize=False, sample_weight=None) 
    print("Without norm acc: {}".format(acc))
    
    #print("\n")
    auc = roc_auc_score(y_true, rounded_pred, average='macro', sample_weight=None) 
    print("roc auc score: {:.6f}".format(auc))
    mcc = matthews_corrcoef(y_true, rounded_pred, sample_weight=None)
    print("Matthews Correlation Coeficient: {:.6f}".format(mcc))
    cohen_kappa = cohen_kappa_score(y_true, rounded_pred, labels=labels, weights=None, sample_weight=None)
    print("cohen_kappa_score: {:.6f} ".format(cohen_kappa))
    
    print("\n")
    print("length of rounded_pred_model: ", len(rounded_pred))
    print("rounded_pred_model: ", rounded_pred)
    print("            y_true: ", y_true)
    #print("\n")
    
    cm = confusion_matrix(y_true, rounded_pred)
    cm_plot_labels = ['Noneoplasico', 'Neoplasico']
    
    plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=False, title='Training Confusion Matrix')
    #print("\n")
    plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=True, title='Training Confusion Matrix')
    print("\n")
    
    target_names = ['No-Neoplasicos', 'Neoplasicos']   #target_names = ['class 0', 'class 1', 'class 2']
    #print(classification_report(y_true, rounded_pred, target_names=target_names))
    print(classification_report_imbalanced(y_true, rounded_pred, target_names=target_names))
    print("\n")

    return f2, acc

