from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import time
import sys
import gc
import numpy as np
import keras

from config.configuration import Configuration
from keepunbalanced import Keep_Unbalanced
from tools.optimazer_factory import Optimizer_Factory
from models.vgg import VGG
from models.resnet import myResNet50
from tools.Dataset_Generator import Dataset_Generator
from tools.callbacks_factory import Callbacks_Factory
from keras import backend as K
from utils import plot_confusion_matrix, print_stats
from sklearn.metrics import confusion_matrix
from bounding_box import BBox

from keras.callbacks import Callback
from sklearn.metrics import recall_score, precision_score, fbeta_score, f1_score, cohen_kappa_score, average_precision_score, precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.utils.multiclass import unique_labels


#def new_session():
#    if K.backend() == 'tensorflow':  # pragma: no cover
#        import tensorflow as tf
#        K.clear_session()
#        config = tf.ConfigProto(allow_soft_placement=True)
#        config.gpu_options.allow_growth = True
#        session = tf.Session(config=config)
#        K.set_session(session)


#class Metrics(Callback):
#    def on_train_begin(self, logs={}):
#        self._data = []
#
#    def on_epoch_end(self, batch, logs={}):
#        X_val, y_val = self.validation_data[0], self.validation_data[1]
#        y_predict = np.asarray(model.predict(X_val))
#
#        y_val = np.argmax(y_val, axis=1)
#        y_predict = np.argmax(y_predict, axis=1)
#
#        self._data.append({
#            'val_recall': recall_score(y_val, y_predict),
#            'val_precision': precision_score(y_val, y_predict),
#        })
#        return
#
#    def get_data(self):
#        return self._data
#

 
class Metrics(keras.callbacks.Callback):
    def __init__(self, model, data_path, data_path2, weights_path, t_gen):
        self._data = []
        self.weights_path = weights_path
        self.model = model
        self.t_gen = t_gen
        self.validation_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                 n_classes=cf.num_classes,
                                                 batch_size=cf.batch_size_valid,
                                                 resize_image=cf.resize_image,
                                                 shuffle_dataset=cf.shuffle_valid,
                                                 seed=cf.seed_valid,
                                                 shuffle_batch=cf.batch_shuffle_valid,
                                                 seed_batch=cf.seed_batch_valid,
                                                 apply_augmentation=False,
                                                 sampling_score=None,
                                                 data_path=data_path,
                                                 data_path2=data_path2,
                                                 mode='validation')



    def on_train_begin(self, logs={}):
        #self.aucs = []
        #self.losses = []
        self.f2_list = [-1]
        self.acc_list = [-1]
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        #self.losses.append(logs.get('loss'))

        #if True:
        #    self.t_gen.mix()

        # predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
        y_pred = self.model.predict_generator(generator=self.validation_generator,
                                              steps=(self.validation_generator.total_images // cf.batch_size_valid),
                                              max_queue_size=1, workers=1)

        rounded_pred = np.argmax(y_pred, axis=1)
        steps = (self.validation_generator.total_images // cf.batch_size_valid)
        y_fnames = self.validation_generator.X_global[0:steps * cf.batch_size_valid]
        y_true = self.validation_generator.y_global[0:steps * cf.batch_size_valid]

        print("> y_fnames, y_true")
        for i in range(len(y_fnames)):
            print("{} {}".format(y_fnames[i], y_true[i]))

        print("> y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 1 predicted ok or 0 other case.")
        for i in range(len(y_pred)):
            if rounded_pred[i] == y_true[i]:
                print("{} {} {} {} {}".format(y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 1))
            else:
                print("{} {} {} {} {}".format(y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 0))


        sys.stdout.flush()


        f2, acc = print_stats(y_true, rounded_pred, epoch=epoch)

        #labels = unique_labels(y_true, rounded_pred)
        #sample_weight = None

        ## ensure, that y_true has at least one 1, because sklearn's fbeta can't handle all-zeros
        ##y_true[:, 0] += 1 - y_true.sum(axis=1).clip(0, 1)


        #p, r, f1, s = precision_recall_fscore_support(y_true, rounded_pred, labels=labels, average=None, sample_weight=sample_weight)

        ##print(">>>>>>>>>>>>>>>>>>>>> ", f1)
        ##print(">>>>>>>>>>>>>>>>>>>>> ", type(f1))
        ##print(">>>>>>>>>>>>>>>>>>>>> ", s)
        ##print(">>>>>>>>>>>>>>>>>>>>> ", np.average(f1, weights=s))

        #beta=2

        #f2_class0 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=0, average='binary', sample_weight=None)
        #f2_class1 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=1, average='binary', sample_weight=None)

        ## f2 score averaged
        #f2 = np.average(np.array([f2_class0, f2_class1]), weights=s)
        #print("\n")
        #print("In epoch: {}".format(epoch+1))
        #print("f2-score: {:.6f} ".format(f2))

        ##return the fraction of correctly classified samples
        #acc_norm = accuracy_score(y_true, rounded_pred, normalize=True, sample_weight=None)
        #print("Normalized   acc: {:.5f}".format(acc_norm))

        ## return the number of correctly classified sample
        #acc = accuracy_score(y_true, rounded_pred, normalize=False, sample_weight=None) 
        #print("Without norm acc: {}".format(acc))

        ##print("\n")
        #auc = roc_auc_score(y_true, rounded_pred, average='macro', sample_weight=None) 
        #print("roc auc score = ", auc)
        #mcc = matthews_corrcoef(y_true, rounded_pred, sample_weight=None)
        #print("Matthews Correlation Coeficient: {:.6f}".format(mcc))
        #cohen_kappa = cohen_kappa_score(y_true, rounded_pred, labels=labels, weights=None, sample_weight=None)
        #print("cohen_kappa_score: {:.6f} ".format(cohen_kappa))

        #print("\n")
        #print("length of rounded_pred_model: ", len(rounded_pred))
        #print("rounded_pred_model: ", rounded_pred)
        #print("            y_true: ", y_true)
        ##print("\n")

        #cm = confusion_matrix(y_true, rounded_pred)
        #cm_plot_labels = ['Noneoplasico', 'Neoplasico']

        #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=False, title='Training Confusion Matrix')
        ##print("\n")
        #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=True, title='Training Confusion Matrix')
        #print("\n")

        #target_names = ['No-Neoplasicos', 'Neoplasicos']   #target_names = ['class 0', 'class 1', 'class 2']
        ##print(classification_report(y_true, rounded_pred, target_names=target_names))
        #print(classification_report_imbalanced(y_true, rounded_pred, target_names=target_names))
        #print("\n")

        ######################
        #if max(self.f2_list) < f2 and max(self.acc_list) < acc:
        #    if cf.checkpoint_enabled and 6 < (epoch+1):
        #        print('\n > On-epoch-end: Saving the model, in epoch {}, to {} '.format(epoch+1, self.weights_path))
        #        self.model.save_weights(self.weights_path)
        
        #if 6 < (epoch+1):   
        #    # append the current values
        #    self.f2_list.append(f2)
        #    self.acc_list.append(acc)


        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

    def get_data(self):
        return self._data


# Create mode, optimizer and callbacks for train or test
def make (cf):
    # Create the optimizer
    print('\n > Creating optimizer...')
    optimizer = Optimizer_Factory().make(cf)

    # Build model
    print('\n > Building model...')
    if cf.model_name == 'vgg16':
        model = VGG(cf.num_classes, optimizer).build_vgg(cf, img_rows=224, img_cols=224, input_channels=3,
                                                         n_layers=16)
    elif cf.model_name == 'resnet50':
        model = myResNet50(cf.num_classes, optimizer).build_resnet50(cf, img_rows=224, img_cols=224, input_channels=3, 
                                                                     load_pretrained=cf.load_imageNet, 
                                                                     freeze_layers_from=cf.freeze_layers_from)
    else:
        raise ValueError('Unknow model')

    return model, optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Configuration file')
    parser.add_argument('-e', '--experiment_num', type=int, default=None, help='Configuration file')
    parser.add_argument('-k', '--kfold', type=int, default=None, help='Configuration file')
    parser.add_argument('-a', '--action', type=str, default=None, help='bbox, divide, train or test')
    args = parser.parse_args()

    cf = Configuration(args.config_path, args.action).load()


    #if args.experiment_num is not None:
    #    print("number of experiment: ", args.experiment_num)
        # TODO: handle the option for a particular train or test


    # Generate images from the bounding boxes of each corresponding mask
    if args.action == 'bbox':
        print("bounding box...")
        bbox = BBox()
        bbox.load(cf)
        bbox.make(cf)

    # Divide the Dataset for Cross-Validation
    if args.action == 'divide':
        print("dividing...")
        if cf.dataset_division_strategy == 'keep-unbalanced':
            for num_experim in range(cf.num_repetition_experiment):
                print('\n > Division: experiment' + str(num_experim))
                ku = Keep_Unbalanced(num_experim)
                ku.load(cf)
                ku.compute_rate(cf)
                ku.separate_by_class(cf)
                ku.make(cf)

    # Train and Validate
    if args.action == 'train':
        sys.stdout.flush()
        print("training...")
        sys.stdout.flush()

        for e in range(cf.num_repetition_experiment):
            # /home/willytell/Experiments/exp1/experiment0_dataset0_22_5_kfold0
            # e: number of experiment
            #print("\n > New experiment...")
            #print("   experiment" + str(e))

            if (args.experiment_num is not None) and (e != args.experiment_num):
                continue
            

            experiment_start_time = time.time()
            #print("cf.num_repetition_experiment = ", cf.num_repetition_experiment)
            #print("e =", e)
            for k in range(cf.n_splits):
                # k: number of kfold
                #print("cf.n_splits = ", cf.n_splits)
                #print("k = ", k)
                #print("\n > New kfold...")
                #print("   kfold" + str(k))

                if (args.kfold is not None) and (k != args.kfold):
                    continue

                full_name_experiment = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                         str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                print("\n ---> Init experiment: " + full_name_experiment + " <---")

                kfold_start_time = time.time()
                print("\n")
                #K.clear_session()
                #tf_session = K.get_session()

                # REPLACED BY make()
                # Create the optimizer
                # print('\n > Creating optimizer...')
                # optimizer = Optimizer_Factory().make(cf)
                #
                # # Build model
                # print('\n > Building model...')
                # if cf.model_name == 'vgg16':
                #     model = VGG(cf.num_classes, optimizer).build_vgg(cf, img_rows=224, img_cols=224, input_channels=3,
                #                                                        n_layers=16)
                # elif cf.model_name == 'resnet50':
                #     model = ResNet50(cf.num_classes, optimizer).build_resnet50(cf, img_rows=224, img_cols=224, input_channels=3)
                # else:
                #     raise ValueError('Unknow model')
                model, optimizer = make(cf)

                # /home/willytell/Experiments/exp1/output/TensorBoard-experiment0/dataset0_22_5_kfold0
                # /home/willytell/Experiments/exp1/output/TensorBoard-experiment0/dataset1_22_5_kfold1
                # /home/willytell/Experiments/exp1/output/TensorBoard-experiment0/dataset2_22_5_kfold2
                # /home/willytell/Experiments/exp1/output/TensorBoard-experiment0/dataset3_22_5_kfold3
                # /home/willytell/Experiments/exp1/output/TensorBoard-experiment0/dataset4_22_5_kfold4
                tensorboard_path = os.path.join(cf.experiments_path, cf.experiment_name,
                                                cf.model_output_directory, 'TensorBoard-' + cf.experiment_prefix + str(e),
                                                cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                                                str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k))

                #/imatge/mgorriz/work/Guillermo/Experiments/exp1-whole_image/vgg-from-scratch
                modelcheckpoint_path = os.path.join(cf.experiments_path, cf.experiment_name, cf.model_output_directory)

                # experiment0_dataset0_24_2_kfold0_weights
                modelcheckpoint_fname = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' + \
                                        str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k) \
                                        + '_' + cf.checkpoint_filename   #'weights-{epoch:02d}-{val_acc:.2f}.hdf5' #'weights.hdf5'

                # Create the callbacks
                print('\n > Creating callbacks...')
                cb = Callbacks_Factory().make(cf, tensorboard_path, modelcheckpoint_path, modelcheckpoint_fname)
                #metrics=Metrics()
                #cb += [metrics]

                data_path = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) + \
                            '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                            str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                data_path2 = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) +                                                                  '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' +                                                                      str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)


                # /home/willytell/Experiments/exp1/output/experiment0_dataset0_22_5_kfold0_weights.hdf5
                weights_path = os.path.join(cf.experiments_path, cf.experiment_name, cf.model_output_directory) + '/' +                                                                   cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                      str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix +                                                             str(k) + '_' + 'on_epoch_end_' + cf.weights_suffix

                # Create the data generators
                train_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                    n_classes=cf.num_classes,
                                                    batch_size=cf.batch_size_train,
                                                    resize_image=cf.resize_image,
                                                    shuffle_dataset=cf.shuffle_train,
                                                    seed=cf.seed_train,
                                                    shuffle_batch=cf.batch_shuffle_train,
                                                    seed_batch=cf.seed_batch_train,
                                                    apply_augmentation=cf.apply_augmentation,
                                                    sampling_score=None,
                                                    data_path=data_path,
                                                    data_path2=data_path2,
                                                    mode='train')

                metrics = Metrics(model, data_path, data_path2, weights_path, train_generator)
                cb += [metrics]

                validation_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                         n_classes=cf.num_classes,
                                                         batch_size=cf.batch_size_valid,
                                                         resize_image=cf.resize_image,
                                                         shuffle_dataset=cf.shuffle_valid,
                                                         seed=cf.seed_valid,
                                                         shuffle_batch=cf.batch_shuffle_valid,
                                                         seed_batch=cf.seed_batch_valid,
                                                         apply_augmentation=False,
                                                         sampling_score=None,
                                                         data_path=data_path,
                                                         data_path2=data_path2,
                                                         mode='validation')

                if cf.apply_augmentation:
                    N = cf.n_augmentation
                else:
                    N = 1
                print('\n > Training the model...')

                class_weights = None #{0: 1.72289156627, 1:1.}
                print('\n > class weights: ', class_weights)
                print('\n')

                history = model.fit_generator(generator=train_generator, validation_data=validation_generator, 
                                              validation_steps=(validation_generator.total_images // cf.batch_size_valid),
                                              class_weight=class_weights,
                                              max_queue_size=5, use_multiprocessing=True, workers=2,
                                              steps_per_epoch=((train_generator.total_images*N) // cf.batch_size_train), 
                                              epochs=cf.n_epochs, verbose=1, callbacks=cb)

                #print("   history: ")
                #print(history.history.keys())
                #print(history.history)

                #print("\n")
                #print("metrics.get_data() = ",metrics.get_data())
                #print("\n")
                #print("metrics.losses = ", metrics.losses)
                #print("metrics.aucs = ", metrics.aucs)

                # /home/willytell/Experiments/exp1/output/experiment0_dataset0_22_5_kfold0_weights.hdf5
                weights_path = os.path.join(cf.experiments_path, cf.experiment_name, cf.model_output_directory) + '/' +                                                                   cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                      str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix +                                                             str(k) + '_' + cf.weights_suffix


                # save the current weights if it wasn't saved any model before.
                if not cf.checkpoint_enabled:
                    print('\n > Saving the model to ', weights_path)
                    model.save_weights(weights_path)


                ###################################################################
 
                # TODO: take the best model and validate (again) it. Review this part of code.

                # it must load the best model!

                #validation_start_time = time.time()
                #print('\n > Validating the model... using validatin set')
                ## evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                #score = model.evaluate_generator(generator=validation_generator,
                #                                 steps=(validation_generator.total_images // cf.batch_size_test))  # , \
                ## max_queue_size=10, \
                ## workers=1, \
                ## use_multiprocessing=False)
                #validation_elapsed_time = time.time() - validation_start_time

                #FPS = validation_generator.total_images / validation_elapsed_time
                #SPF = validation_elapsed_time / validation_generator.total_images
                #print("   Validation time: {:.11f}. FPS: {:.11f}. Seconds per Frame: {:.11f}".format(
                #    validation_elapsed_time, FPS, SPF))
                #print("   Validation metrics:")
                #print("      acc: {:.6f}, ".format(score[1]))
                #print("      loss: {:.12f}".format(score[0]))
                ## print("   Loss: ", score[0], "Accuracy: ", score[1])
                #print("\n")


                ## predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                #predictions = model.predict_generator(generator=validation_generator,
                #                                      steps=(validation_generator.total_images // cf.batch_size_valid))

                #print("predictions = ", predictions)
                #print ("np.argmax(predicitons) = ", np.argmax(predictions, axis=1))
                #print("\n")

                #rounded_pred_model = np.array([], dtype=np.int64)
                #for p in predictions:
                #    rounded_pred_model = np.append(rounded_pred_model, np.argmax(p))

                #print("length of rounded_pred_model = ", len(rounded_pred_model))
                #print("rounded_pred_model = ", rounded_pred_model)

                ## print("len(predict_generator.history_batch_labels) = ", len(predict_generator.history_batch_labels))
                ## print("predict_generator.history_batch_labels = ", predict_generator.history_batch_labels)

                #steps = (validation_generator.total_images // cf.batch_size_valid)
                #y_true = validation_generator.history_batch_labels[0:steps * cf.batch_size_valid]
                ## y_true = predict_generator.history_batch_labels[0:len(predict_generator.history_batch_labels)-cf.batch_size_test]
                ## print("len(y_true) = ", len(y_true))
                #print("            y_true = ", y_true)
                #print("\n")

                #cm = confusion_matrix(y_true, rounded_pred_model)
                #cm_plot_labels = ['Noneoplasico', 'Neoplasico']

                #fname = os.path.join(cf.experiments_path, cf.experiment_name,
                #                     cf.model_output_directory) + '/' + cf.experiment_prefix + str(e) + \
                #        '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                #        str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k) + '_' #+ 'cmatrix_validation.jpg'

                ##plot_confusion_matrix(cm, cm_plot_labels, fname + 'cmatrix_validation.jpg', normalize=False, title='Confusion Matrix')
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=False, title='Confusion Matrix')
                #print("\n")
                ##plot_confusion_matrix(cm, cm_plot_labels, fname + 'cmatrix_normalized_validation.jpg', normalize=True, title='Confusion Matrix')
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=True, title='Confusion Matrix')

                #target_names = ['No-Neoplasicos', 'Neoplasicos']   #target_names = ['class 0', 'class 1', 'class 2']
                #print(classification_report(y_true, rounded_pred_model, target_names=target_names))

                ###################################################################



                print('\n > Deleting the model.',)
                #tf_session.clear_session()  
                print('\n')            
                sys.stdout.flush()
                K.clear_session()
                sys.stdout.flush()
                del model, train_generator, validation_generator, optimizer, cb, metrics.validation_generator, metrics
                gc.collect()


                kfold_elapsed_time = time.time() - kfold_start_time
                print("\n > Finished experiment " + full_name_experiment + " in {:.2f} seconds.".format(kfold_elapsed_time))

                print("\n ---> Finish experiment: " + full_name_experiment + " <---")



            experiment_elapsed_time = time.time() - experiment_start_time
            print("\n > Time to compute the experiment{}: {:.2f} seconds.".format(e, experiment_elapsed_time))

    if args.action == 'test':
        print("testing...")

        for e in range(cf.num_repetition_experiment):
            # /home/willytell/Experiments/exp1/experiment0_dataset0_22_5_kfold0
            # e: number of experiment
            #print("\n > New experiment...")
            #print("   experiment" + str(e))
            
            if (args.experiment_num is not None) and (e != args.experiment_num):
                continue
            

            experiment_start_time = time.time()
            #print("cf.num_repetition_experiment = ", cf.num_repetition_experiment)
            #print("e =", e)
            for k in range(cf.n_splits):
                # k: number of kfold
                #print("cf.n_splits = ", cf.n_splits)
                #print("k = ", k)
                #print("\n > New kfold...")
                #print("   kfold" + str(k))

                if (args.kfold is not None) and (k != args.kfold):
                    continue


                full_name_experiment = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                         str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                print("\n ---> Init experiment: " + full_name_experiment + " <---")


                #print("\n")
                #K.clear_session()

                model, optimizer = make(cf)

                # /home/willytell/Experiments/exp1/output/experiment0_dataset0_22_5_kfold0_weights.hdf5
                weights_path = os.path.join(cf.experiments_path, cf.experiment_name,
                                            cf.model_output_directory) + '/' + cf.experiment_prefix + str(e) + \
                               '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                               str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k) + '_' + cf.weights_suffix

                print('\n > Loading the model from ', weights_path)
                model.load_weights(weights_path, by_name=False)


                data_path = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix +                                                                           str(e) + '_' + cf.dataset_prefix + '_' + str(cf.num_images_for_test) + '_' +                                                                      str(cf.n_splits) 

                data_path2 = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) +                                                                  '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' +                                                                      str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                test_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                   n_classes=cf.num_classes,
                                                   batch_size=cf.batch_size_test,
                                                   resize_image=cf.resize_image,
                                                   shuffle_dataset=cf.shuffle_test,
                                                   seed=cf.seed_test,
                                                   shuffle_batch=cf.batch_shuffle_test,
                                                   seed_batch=cf.seed_batch_test,
                                                   apply_augmentation=False,
                                                   sampling_score=None,
                                                   data_path=data_path,
                                                   data_path2=data_path2,
                                                   mode='test')

                predict_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                   n_classes=cf.num_classes,
                                                   batch_size=cf.batch_size_test,
                                                   resize_image=cf.resize_image,
                                                   shuffle_dataset=cf.shuffle_test,
                                                   seed=cf.seed_test,
                                                   shuffle_batch=cf.batch_shuffle_test,
                                                   seed_batch=cf.seed_batch_test,
                                                   apply_augmentation=False,
                                                   sampling_score=None,
                                                   data_path=data_path,
                                                   data_path2=data_path2,
                                                   mode='test')


                # classes_generator = Dataset_Generator(cf, cf.dataset_images_path,
                #                                    n_classes=cf.num_classes,
                #                                    batch_size=cf.batch_size_test,
                #                                    resize_image=cf.resize_image,
                #                                    flag_shuffle=cf.shuffle_test,
                #                                    apply_augmentation=False,
                #                                    sampling_score=None,
                #                                    data_path=data_path,
                #                                    data_path2=data_path2,
                #                                    mode='test')


                ###################################################################

                # data_path91 = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)
                #
                # data_path92 = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)
                #
                # validation_generator = Dataset_Generator(cf, cf.dataset_images_path,
                #                                          n_classes=cf.num_classes,
                #                                          batch_size=cf.batch_size_valid,
                #                                          resize_image=cf.resize_image,
                #                                          flag_shuffle=cf.shuffle_valid,
                #                                          apply_augmentation=False,
                #                                          sampling_score=None,
                #                                          data_path=data_path91,
                #                                          data_path2=data_path92,
                #                                          mode='validation')
                #
                # validation_start_time = time.time()
                # print('\n > Validating the model... using validatin set')
                # # evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                # score = model.evaluate_generator(generator=validation_generator,
                #                                  steps=(validation_generator.total_images // cf.batch_size_test))  # , \
                # # max_queue_size=10, \
                # # workers=1, \
                # # use_multiprocessing=False)
                # validation_elapsed_time = time.time() - validation_start_time
                # FPS = validation_generator.total_images / validation_elapsed_time
                # SPF = validation_elapsed_time / validation_generator.total_images
                # print("   Validation time: {:.11f}. FPS: {:.11f}. Seconds per Frame: {:.11f}".format(validation_elapsed_time, FPS, SPF))
                # print("   Validation metrics:")
                # print("      acc: {:.6f}, ".format(score[1]))
                # print("      loss: {:.12f}".format(score[0]))
                # # print("   Loss: ", score[0], "Accuracy: ", score[1])
                # print("\n")

                ###################################################################

                test_start_time = time.time()
                print('\n > Testing the model...')
                # evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                score = model.evaluate_generator(generator=test_generator,
                                                 steps=(test_generator.total_images // cf.batch_size_test)) #, \
                                         #max_queue_size=10, \
                                         #workers=1, \
                                         #use_multiprocessing=False)
                test_elapsed_time = time.time() - test_start_time
                FPS = test_generator.total_images/test_elapsed_time
                SPF = test_elapsed_time/test_generator.total_images
                print("   Testing time: {:.11f}. FPS: {:.11f}. Seconds per Frame: {:.11f}".format(test_elapsed_time, FPS, SPF))
                print("   Test metrics:")
                print("      acc: {:.6f} ".format(score[1]))
                print("      loss: {:.12f}".format(score[0]))
                #print("   Loss: ", score[0], "Accuracy: ", score[1])
                print("\n")

                #predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                y_pred = model.predict_generator(generator=predict_generator,
                                                      steps=(predict_generator.total_images // cf.batch_size_test))

                rounded_pred = np.argmax(y_pred, axis=1)
                steps = (predict_generator.total_images // cf.batch_size_test)
                y_fnames = predict_generator.X_global[0:steps * cf.batch_size_test]
                #print("\ny_fnames = ")
                #print(y_fnames)

                #print("\ny_pred = ") 
                #print(y_pred)

                #print ("np.argmax(y_pred) = ", np.argmax(y_pred))
                #print("\n")

                #rounded_pred = np.array([], dtype=np.int64)
                #for p in y_pred:
                #    rounded_pred=np.append(rounded_pred, np.argmax(p))

                print ("rounded_pred = ", rounded_pred)


                y_true = predict_generator.y_global[0:steps*cf.batch_size_test]
                #print("len(y_true) = ", len(y_true))
                print("       y_true = ", y_true)
                print("\n")

                print("> y_fnames, y_true")
                for i in range(len(y_fnames)):
                    print("{} {}".format(y_fnames[i], y_true[i]))
        
                print("\n> y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 1 predicted ok or 0 other case.")
                for i in range(len(y_pred)):
                    if rounded_pred[i] == y_true[i]:
                        print("{} {} {} {} {}".format(y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 1)) 
                    else:
                        print("{} {} {} {} {}".format(y_pred[i,0], y_pred[i,1], abs(y_pred[i,0] - y_pred[i,1]), rounded_pred[i], 0)) 
        
        
                sys.stdout.flush()


                ######################################################################3 
                
                print_stats(y_true, rounded_pred)
 
                #labels = unique_labels(y_true, rounded_pred)
                #sample_weight = None
        
                ## ensure, that y_true has at least one 1, because sklearn's fbeta can't handle all-zeros
                ##y_true[:, 0] += 1 - y_true.sum(axis=1).clip(0, 1)
        
        
                #p, r, f1, s = precision_recall_fscore_support(y_true, rounded_pred, labels=labels, average=None, sample_weight=sample_weight)
        
                ##print(">>>>>>>>>>>>>>>>>>>>> ", f1)
                ##print(">>>>>>>>>>>>>>>>>>>>> ", type(f1))
                ##print(">>>>>>>>>>>>>>>>>>>>> ", s)
                ##print(">>>>>>>>>>>>>>>>>>>>> ", np.average(f1, weights=s))
        
                #beta=2
        
                #f2_class0 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=0, average='binary', sample_weight=None)
                #f2_class1 = fbeta_score(y_true, rounded_pred, beta=beta, labels=labels, pos_label=1, average='binary', sample_weight=None)
        
                ## f2 score averaged
                #f2 = np.average(np.array([f2_class0, f2_class1]), weights=s)
                #print("\n")
                ##print("In epoch: {}".format(epoch+1))
                #print("f2-score: {:.6f} ".format(f2))
        
                ##return the fraction of correctly classified samples
                #acc_norm = accuracy_score(y_true, rounded_pred, normalize=True, sample_weight=None)
                #print("Normalized   acc: {:.5f}".format(acc_norm))
        
                ## return the number of correctly classified sample
                #acc = accuracy_score(y_true, rounded_pred, normalize=False, sample_weight=None) 
                #print("Without norm acc: {}".format(acc))
        
                ##print("\n")
                #auc = roc_auc_score(y_true, rounded_pred, average='macro', sample_weight=None) 
                #print("roc auc score = ", auc)
                #mcc = matthews_corrcoef(y_true, rounded_pred, sample_weight=None)
                #print("Matthews Correlation Coeficient: {:.6f}".format(mcc))
                #cohen_kappa = cohen_kappa_score(y_true, rounded_pred, labels=labels, weights=None, sample_weight=None)
                #print("cohen_kappa_score: {:.6f} ".format(cohen_kappa))
        
                #print("\n")
                #print("length of rounded_pred_model: ", len(rounded_pred))
                #print("rounded_pred_model: ", rounded_pred)
                #print("            y_true: ", y_true)
                ##print("\n")
        
                #cm = confusion_matrix(y_true, rounded_pred)
                #cm_plot_labels = ['Noneoplasico', 'Neoplasico']
        
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=False, title='Training Confusion Matrix')
                ##print("\n")
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=True, title='Training Confusion Matrix')
                #print("\n")
        
                #target_names = ['No-Neoplasicos', 'Neoplasicos']   #target_names = ['class 0', 'class 1', 'class 2']
                ##print(classification_report(y_true, rounded_pred, target_names=target_names))
                #print(classification_report_imbalanced(y_true, rounded_pred, target_names=target_names))
                #print("\n")


                #######################################################################################3


                #cm = confusion_matrix(y_true, rounded_pred_model)
                #cm_plot_labels = ['Noneoplasico','Neoplasico']

                #fname = os.path.join(cf.experiments_path, cf.experiment_name,
                #                            cf.model_output_directory) + '/' + cf.experiment_prefix + str(e) + \
                #               '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                #               str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k) + '_' #+ 'cmatrix_testing.jpg'

                ##plot_confusion_matrix(cm, cm_plot_labels, fname + 'cmatrix_testing.jpg', normalize=False, title='Confusion Matrix')
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=False, title='Testing Confusion Matrix')
                #print("\n")
                ##plot_confusion_matrix(cm, cm_plot_labels, fname + 'cmatrix_normalized_testing.jpg', normalize=True, title='Confusion Matrix')
                #plot_confusion_matrix(cm, cm_plot_labels, fname=None, normalize=True, title='Testing Confusion Matrix')

                ##rounded_predictions = model.predict_classes(classes_generator, 
                ##                                            batch_size=(predict_generator.total_images// cf.batch_size_test), verbose=0)

                ##print ("rounded_predictions = ", rounded_predictions)


                print('\n > Deleting the model.',)
                print("\n")
                sys.stdout.flush()
                K.clear_session()
                sys.stdout.flush()

                del model, test_generator, predict_generator, optimizer # , classes_generator
                gc.collect()


                print("\n ---> Finish experiment: " + full_name_experiment + " <---")


    print("\n bye bye!")
