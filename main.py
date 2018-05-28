from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import time
import gc
import numpy as np

from config.configuration import Configuration
from keepunbalanced import Keep_Unbalanced
from tools.optimazer_factory import Optimizer_Factory
from models.vgg import VGG
from models.resnet import myResNet50
from tools.Dataset_Generator import Dataset_Generator
from tools.callbacks_factory import Callbacks_Factory
from keras import backend as K
from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from bounding_box import BBox

#def new_session():
#    if K.backend() == 'tensorflow':  # pragma: no cover
#        import tensorflow as tf
#        K.clear_session()
#        config = tf.ConfigProto(allow_soft_placement=True)
#        config.gpu_options.allow_growth = True
#        session = tf.Session(config=config)
#        K.set_session(session)


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
        model = myResNet50(cf.num_classes, optimizer).build_resnet50(cf, img_rows=224, img_cols=224, input_channels=3)
    else:
        raise ValueError('Unknow model')

    return model, optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Configuration file')
    parser.add_argument('-ne', '--number_of_experiment', type=int, default=None, help='Configuration file')
    parser.add_argument('-a', '--action', type=str, default=None, help='bbox, divide, train or test')
    args = parser.parse_args()

    cf = Configuration(args.config_path, args.action).load()


    if args.number_of_experiment is not None:
        print("number of experiment: ", args.number_of_experiment)
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
        print("training...")

        for e in range(cf.num_repetition_experiment):
            # /home/willytell/Experiments/exp1/experiment0_dataset0_22_5_kfold0
            # e: number of experiment
            #print("\n > New experiment...")
            #print("   experiment" + str(e))
            experiment_start_time = time.time()
            #print("cf.num_repetition_experiment = ", cf.num_repetition_experiment)
            #print("e =", e)
            for k in range(cf.n_splits):
                # k: number of kfold
                #print("cf.n_splits = ", cf.n_splits)
                #print("k = ", k)
                #print("\n > New kfold...")
                #print("   kfold" + str(k))

                full_name_experiment = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                         str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                print("\n ---> Init experiment: " + full_name_experiment + " <---")

                kfold_start_time = time.time()
                print("\n")
                K.clear_session()
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
                                        + '_' + 'weights.hdf5'

                # Create the callbacks
                print('\n > Creating callbacks...')
                cb = Callbacks_Factory().make(cf, tensorboard_path, modelcheckpoint_path, modelcheckpoint_fname)

                data_path = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) + \
                            '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                            str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                data_path2 = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(e) +                                                                  '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' +                                                                      str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)


                # Create the data generators
                train_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                    n_classes=cf.num_classes,
                                                    batch_size=cf.batch_size_train,
                                                    resize_image=cf.resize_image,
                                                    shuffle=cf.shuffle_train,
                                                    apply_augmentation=False,
                                                    sampling_score=None,
                                                    data_path=data_path,
                                                    mode='train')

                validation_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                         n_classes=cf.num_classes,
                                                         batch_size=cf.batch_size_valid,
                                                         resize_image=cf.resize_image,
                                                         shuffle=cf.shuffle_valid,
                                                         apply_augmentation=False,
                                                         sampling_score=None,
                                                         data_path=data_path,
                                                         data_path2=data_path2,
                                                         mode='validation')
                print('\n > Training the model...')
                model.fit_generator(generator=train_generator.generate(), validation_data=validation_generator.generate(),
                                    validation_steps=(validation_generator.total_images // cf.batch_size_valid),
                                    steps_per_epoch=(train_generator.total_images // cf.batch_size_train),
                                    epochs=cf.n_epochs, verbose=1, callbacks=cb)



                # /home/willytell/Experiments/exp1/output/experiment0_dataset0_22_5_kfold0_weights.hdf5
                weights_path = os.path.join(cf.experiments_path, cf.experiment_name, cf.model_output_directory) + '/' +                                                                   cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                      str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix +                                                             str(k) + '_' + cf.weights_suffix


                if not cf.checkpoint_enabled:
                    print('\n > Saving the model to ', weights_path)
                    model.save_weights(weights_path)

                print('\n > Deleting the model.',)
                #tf_session.clear_session()  
                del model, train_generator, validation_generator, optimizer, cb
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
            experiment_start_time = time.time()
            #print("cf.num_repetition_experiment = ", cf.num_repetition_experiment)
            #print("e =", e)
            for k in range(cf.n_splits):
                # k: number of kfold
                #print("cf.n_splits = ", cf.n_splits)
                #print("k = ", k)
                #print("\n > New kfold...")
                #print("   kfold" + str(k))

                full_name_experiment = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + str(k) + '_' +                                                                         str(cf.num_images_for_test) + '_' + str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k)

                print("\n ---> Init experiment: " + full_name_experiment + " <---")


                print("\n")
                K.clear_session()

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
                                                   shuffle=cf.shuffle_test,
                                                   apply_augmentation=False,
                                                   sampling_score=None,
                                                   data_path=data_path,
                                                   data_path2=data_path2,
                                                   mode='test')

                predict_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                   n_classes=cf.num_classes,
                                                   batch_size=cf.batch_size_test,
                                                   resize_image=cf.resize_image,
                                                   shuffle=cf.shuffle_test,
                                                   apply_augmentation=False,
                                                   sampling_score=None,
                                                   data_path=data_path,
                                                   data_path2=data_path2,
                                                   mode='test')


                classes_generator = Dataset_Generator(cf, cf.dataset_images_path,
                                                   n_classes=cf.num_classes,
                                                   batch_size=cf.batch_size_test,
                                                   resize_image=cf.resize_image,
                                                   shuffle=cf.shuffle_test,
                                                   apply_augmentation=False,
                                                   sampling_score=None,
                                                   data_path=data_path,
                                                   data_path2=data_path2,
                                                   mode='test')


                test_start_time = time.time()
                print('\n > Testing the model...')
                # evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                score = model.evaluate_generator(generator=test_generator.generate(), \
                                         steps=(test_generator.total_images // cf.batch_size_test)) #, \
                                         #max_queue_size=10, \
                                         #workers=1, \
                                         #use_multiprocessing=False)
                test_elapsed_time = time.time() - test_start_time
                FPS = test_generator.total_images/test_elapsed_time
                SPF = test_elapsed_time/test_generator.total_images
                print("   Testing time: {:.11f}. FPS: {:.11f}. Seconds per Frame: {:.11f}".format(test_elapsed_time, FPS, SPF))
                print("   Test metrics:")
                print("      acc: {:.6f}, ".format(score[1]))
                print("      loss: {:.12f}".format(score[0]))
                #print("   Loss: ", score[0], "Accuracy: ", score[1])
                print("\n")

                #predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
                predictions = model.predict_generator(generator=predict_generator.generate(),
                                                      steps=(predict_generator.total_images // cf.batch_size_test))

                print("predictions = ", predictions)
                #print ("np.argmax(predicitons) = ", np.argmax(predictions))
                print("\n")

                rounded_pred_model = np.array([], dtype=np.int64)
                for p in predictions:
                    rounded_pred_model=np.append(rounded_pred_model, np.argmax(p))

                print ("rounded_pred_model = ", rounded_pred_model)


                #print("len(predict_generator.history_batch_labels) = ", len(predict_generator.history_batch_labels))
                #print("predict_generator.history_batch_labels = ", predict_generator.history_batch_labels)

                y_true = predict_generator.history_batch_labels[0:len(predict_generator.history_batch_labels)-cf.batch_size_test]
                #print("len(y_true) = ", len(y_true))
                print("            y_true = ", y_true)
                print("\n")

                cm = confusion_matrix(y_true, rounded_pred_model)
                cm_plot_labels = ['Noneoplasico','Neoplasico']

                fname = os.path.join(cf.experiments_path, cf.experiment_name,
                                            cf.model_output_directory) + '/' + cf.experiment_prefix + str(e) + \
                               '_' + cf.dataset_prefix + str(k) + '_' + str(cf.num_images_for_test) + '_' + \
                               str(cf.n_splits) + '_' + cf.n_splits_prefix + str(k) + '_' + 'cmatrix.jpg'

                plot_confusion_matrix(cm, cm_plot_labels, fname, title='Confusion Matrix')

                #rounded_predictions = model.predict_classes(classes_generator.generate(), 
                #                                            batch_size=(predict_generator.total_images// cf.batch_size_test), verbose=0)

                #print ("rounded_predictions = ", rounded_predictions)

                print("\n ---> Finish experiment: " + full_name_experiment + " <---")


    print("\n bye bye!")
