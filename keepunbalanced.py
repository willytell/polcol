import os
import sys
import numpy as np
from utils import read_data, rate, filter_by_class, shuffle, dataset_bins_idx

class Keep_Unbalanced(object):
    def __init__(self, number_of_experiment):
        self.number_of_experiment = number_of_experiment
        self.n_samples = 0
        self.n_classes = 0
        self.frequency = 0

    def load(self, cf):
        self.X_original, self.y_original = read_data(cf.dataset_directory, cf.images_filenames, cf.labels)

        # classes = np.unique(y_all) that is equal to ['NONEO', 'NEO']
        self.classes, self.frequency = np.unique(self.y_original, return_counts=True) # count the frequency of unique values

        self.n_samples = len(self.X_original)
        self.n_classes = len(self.classes)

        print("keep_unbalanced classes = ", self.classes)

    def compute_rate(self, cf):
        # Rates of each class
        self.proportion = rate(self.n_samples, self.frequency)
        if cf.num_classes == 2:
            print("self.proportion[0] = ", self.proportion[0])  # NONEO
            print("self.proportion[1] = ", self.proportion[1])  # NEO

    def separate_by_class(self, cf):
        if cf.num_classes == 2:
            self.X_noneo, self.y_noneo = filter_by_class(self.X_original, self.y_original, 'NONEOPLASICO')
            self.X_neop, self.y_neop = filter_by_class(self.X_original, self.y_original, 'NEOPLASICO')

    # Leave One-Out: preparing the Test set
    def leave_one_out(self, cf):
        if cf.num_images_for_test > 0:
            self.test_noneo_images = int(round(cf.num_images_for_test * self.proportion[1]))  # proportion[1] means the rate of the NOneoplasia of 37%
            self.test_neop_images = (cf.num_images_for_test - self.test_noneo_images)
        else:
            # when our test set has 0 images for test, exceptional case.
            self.test_noneo_images = 0
            self.test_neop_images = 0
    
        # test images used from noneo
        self.X_test_noneo = np.copy(self.X_noneo[0:self.test_noneo_images])
        self.y_test_noneo = np.copy(self.y_noneo[0:self.test_noneo_images])
    
        # rest of noneo used for train and validation
        self.X_train_val_noneo = np.copy(self.X_noneo[self.test_noneo_images:])
        self.y_train_val_noneo = np.copy(self.y_noneo[self.test_noneo_images:])
    
    
        # test images used from neop
        self.X_test_neop = np.copy(self.X_neop[0:self.test_neop_images])
        self.y_test_neop = np.copy(self.y_neop[0:self.test_neop_images])
    
        # rest of neop used for train and validation
        self.X_train_val_neop = np.copy(self.X_neop[self.test_neop_images:])
        self.y_train_val_neop = np.copy(self.y_neop[self.test_neop_images:])
    
        # Test set
        self.X_test = np.concatenate((self.X_test_noneo, self.X_test_neop), axis=0)
        self.y_test = np.concatenate((self.y_test_noneo, self.y_test_neop), axis=0)
            



    def make(self, cf):

        if cf.num_classes == 2:
            print('\n > Shuffling the noneo and neop classes...')
            shuffle(self.X_noneo, self.y_noneo)
            shuffle(self.X_neop, self.y_neop)

            # Test set
            print('\n > Test set...')
            self.leave_one_out(cf)

            # Print the X_test and y_test
            print('X_test:')
            print(self.X_test)
            print('y_test:')
            print(self.y_test)
            print('\n')

            cad = cf.experiment_prefix + str(self.number_of_experiment) + '_' + cf.dataset_prefix + '_' + str(cf.num_images_for_test) + '_' + \
                  str(cf.n_splits)

            # Saving the X_test and y_test for Neop and Noneo
            np.save(os.path.join(cf.output_path, cad + '_X_test_neop'), self.X_test_neop)
            np.save(os.path.join(cf.output_path, cad + '_y_test_neop'), self.y_test_neop)

            np.save(os.path.join(cf.output_path, cad + '_X_test_noneo'), self.X_test_noneo)
            np.save(os.path.join(cf.output_path, cad + '_y_test_noneo'), self.y_test_noneo)


            # Generate idx of each bin
            neop_gen_idx = dataset_bins_idx(self.X_train_val_neop, self.y_train_val_neop, cf.n_splits)
            noneo_gen_idx = dataset_bins_idx(self.X_train_val_noneo, self.y_train_val_noneo, cf.n_splits)

            neop_bin_idx = np.array([0], dtype=np.int32)
            noneo_bin_idx = np.array([0], dtype=np.int32)

            # Iterate the generators
            for i, j in zip(neop_gen_idx, noneo_gen_idx):
                neop_bin_idx = np.append(neop_bin_idx, i)
                noneo_bin_idx = np.append(noneo_bin_idx, j)

            assert (len(neop_bin_idx) == len(noneo_bin_idx))



            pick_one_bin_for_validation = 0

            for present_bin in range(len(neop_bin_idx)-1):

                #print("present_bin = ", present_bin)

                self.X_train_neop = np.array([])
                self.y_train_neop = np.array([])
                self.train_neop_bin_idx = np.array([0], dtype=np.int32)

                self.X_train_noneo = np.array([])
                self.y_train_noneo = np.array([])
                self.train_noneo_bin_idx = np.array([0])

                # Initialize variables each time a bin is selected for validation
                self.X_validation_neop = np.array([])
                self.y_validation_neop = np.array([])
                # self.validation_neop_bin_idx = np.array([0])

                self.X_validation_noneo = np.array([])
                self.y_validation_noneo = np.array([])
                # self.validation_noneo_bin_idx = np.array([0])


                #print("len(neop_bin_idx) = ", len(neop_bin_idx))

                idx = 1
                while idx < len(neop_bin_idx):

                    #print("idx = ", idx)
                    #print("pick_one_bin_for_validation = ", pick_one_bin_for_validation)

                    # present bin is picked for validation set
                    if (idx-1) == pick_one_bin_for_validation:

                        # NEO: Validation
                        self.X_validation_neop = np.append(self.X_validation_neop, self.X_train_val_neop[neop_bin_idx[idx-1]:neop_bin_idx[idx]])
                        self.y_validation_neop = np.append(self.y_validation_neop, self.y_train_val_neop[neop_bin_idx[idx-1]:neop_bin_idx[idx]])
                        #self.validation_neop_bin_idx = np.append(self.validation_neop_bin_idx, len(self.X_validation_neop))
                        #print("self.validation_neop_bin_idx = ", self.validation_neop_bin_idx)

                        #print("self.X_validation_neop = ", self.X_validation_neop)
                        #print("self.y_validation_neop = ", self.y_validation_neop)

                        # NONEO Validation
                        self.X_validation_noneo = np.append(self.X_validation_noneo, self.X_train_val_noneo[noneo_bin_idx[idx-1]:noneo_bin_idx[idx]])
                        self.y_validation_noneo = np.append(self.y_validation_noneo, self.y_train_val_noneo[noneo_bin_idx[idx-1]:noneo_bin_idx[idx]])
                        #self.validation_noneo_bin_idx = np.append(self.validation_noneo_bin_idx, len(self.X_validation_noneo))
                        #print("self.validation_noneo_bin_idx = ", self.validation_noneo_bin_idx)
                    else:
                        # present bin must be for train set
                        self.X_train_neop = np.append(self.X_train_neop, self.X_train_val_neop[neop_bin_idx[idx-1]:neop_bin_idx[idx]])
                        self.y_train_neop = np.append(self.y_train_neop, self.y_train_val_neop[neop_bin_idx[idx-1]:neop_bin_idx[idx]])
                        self.train_neop_bin_idx = np.append(self.train_neop_bin_idx, len(self.X_train_neop))
                        #self.train_neop_bin_idx = np.concatenate((self.train_neop_bin_idx, [len(self.X_train_neop)]), axis=0)

                        #print("neop_bin_idx = ", neop_bin_idx)
                        #print("self.train_neop_bin_idx = ", self.train_neop_bin_idx)

                        # present bin must be for train set
                        self.X_train_noneo = np.append(self.X_train_noneo,
                                                      self.X_train_val_noneo[noneo_bin_idx[idx - 1]:noneo_bin_idx[idx]])
                        self.y_train_noneo = np.append(self.y_train_noneo,
                                                      self.y_train_val_noneo[noneo_bin_idx[idx - 1]:noneo_bin_idx[idx]])
                        self.train_noneo_bin_idx = np.append(self.train_noneo_bin_idx, len(self.X_train_noneo))
                        #self.train_noneo_bin_idx = np.concatenate((self.train_noneo_bin_idx, [len(self.X_train_noneo)]), axis=0)

                        #print("noneo_bin_idx = ", noneo_bin_idx)
                        #print("self.train_noneo_bin_idx = ", self.train_noneo_bin_idx)

                        print("   X_train_neop bin size : ", len( self.X_train_val_neop [neop_bin_idx[idx-1]:neop_bin_idx[idx]]))
                        print("   X_train_noneo bin size : ",len(self.X_train_val_noneo[noneo_bin_idx[idx - 1]:noneo_bin_idx[idx]]))

                    idx = idx+1

                # save files

                cad2 = cf.experiment_prefix + str(self.number_of_experiment) + '_' + cf.dataset_prefix + str(pick_one_bin_for_validation) + '_' + str(cf.num_images_for_test) + '_' + \
                      str(cf.n_splits) + '_' + cf.n_splits_prefix + str(pick_one_bin_for_validation) + '_'



                np.save(os.path.join(cf.output_path, cad2 + 'X_validation_neop'), self.X_validation_neop)
                np.save(os.path.join(cf.output_path, cad2 + 'y_validation_neop'), self.y_validation_neop)
                #np.save(os.path.join(cf.output_path, cad2 + 'X_validation_neop_bin_idx'), self.validation_neop_bin_idx)


                np.save(os.path.join(cf.output_path, cad2 + 'X_validation_noneo'), self.X_validation_noneo)
                np.save(os.path.join(cf.output_path, cad2 + 'y_validation_noneo'), self.y_validation_noneo)
                #np.save(os.path.join(cf.output_path, cad2 + 'X_validation_noneo_bin_idx'), self.validation_noneo_bin_idx)



                np.save(os.path.join(cf.output_path, cad2 + 'X_train_neop'), self.X_train_neop)
                np.save(os.path.join(cf.output_path, cad2 + 'y_train_neop'), self.y_train_neop)
                np.save(os.path.join(cf.output_path, cad2 + 'X_train_neop_bin_idx'), self.train_neop_bin_idx)


                np.save(os.path.join(cf.output_path, cad2 + 'X_train_noneo'), self.X_train_noneo)
                np.save(os.path.join(cf.output_path, cad2 + 'y_train_noneo'), self.y_train_noneo)
                np.save(os.path.join(cf.output_path, cad2 + 'X_train_noneo_bin_idx'), self.train_noneo_bin_idx)

                # Printing the division of the dataset
                print('\n   > ' + cad2 + ':')

                print('\n > X_validation_neop')
                print(self.X_validation_neop)
                print('\n > y_validation_neop')
                print(self.y_validation_neop)

                print('\n > X_validation_noneo')
                print(self.X_validation_noneo)
                print('\n > y_validation_noneo')
                print(self.y_validation_noneo)

                print('\n > X_train_neop')
                print(self.X_train_neop)
                print('\n > y_train_neop')
                print(self.y_train_neop)

                print('\n > X_train_noneo')
                print(self.X_train_noneo)
                print('\n > y_train_noneo')
                print(self.y_train_noneo)

                sys.stdout.flush()

                pick_one_bin_for_validation = pick_one_bin_for_validation + 1



