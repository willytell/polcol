import numpy as np
import argparse
import os
import sys
import shutil

from utils import read_data, filter_by_class
from importlib.machinery import SourceFileLoader


# def filter_by_class(X, y, filter):
#     idx = np.where(y == filter)  # filter: 0 = 'NONEOPLASICO' and 1 = 'NEOPLASICO'
#     idx_arr = np.asarray(idx[0])  # tupla to np.array
#
#     return np.copy(X[idx_arr]), np.copy(y[idx_arr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save prefix to image')
    parser.add_argument('-i', '--input_dir', type=str, default=None, help='directory with images')
    parser.add_argument('-d', '--data', type=str, default='data.csv', help='csv file with data')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='directory to save images')
    parser.add_argument('-c', '--column', type=int, default=None, help='column to set the prefix')
    parser.add_argument('-t', '--threshold', type=float, default=None, help='threshols')
    parser.add_argument('-im', '--images', type=str, default=None, help='full path to images list')
    parser.add_argument('-l', '--labels', type=str, default=None, help='full path to labels list')
    parser.add_argument('-config', '--configuration', type=str, default=None, help='configuration file')
    parser.add_argument('-e', '--experiment', type=str, default=None, help='experiment number')
    parser.add_argument('-k', '--kfold', type=str, default=None, help='kfold number')
    parser.add_argument('-a', '--action', type=str, default=None, help='feed_cnn, prefix, select, test')
    args = parser.parse_args()

    # Example:
    # Option -c: -c 1 = "% Aciertos", -c 2 = "Mean (Diff)", -c 3 = "DevStd (Diff)"
    # python save_prefix_image.py -i "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox" -d "data.csv" -c 1 -o "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/test" -a "prefix"
    if args.action == 'prefix':
        #"Image Name", "class", "% Aciertos", "Mean (Diff)", "DevStd (Diff)"
        image_names = np.genfromtxt(args.data, delimiter=",", usecols=(0), dtype='str')
        stats = np.genfromtxt(args.data, delimiter=",", usecols=(1,2,3,4))

        # We have only 2 classes
        # noneo = class 0
        # neo   = class 1

        for idx, fname in enumerate(image_names):
            old_name = image_names[idx]
            source = os.path.join(args.input_dir, old_name)
            print("source: {}".format(source))

            # args.column_prefix determines the column to be used to set the prefix
            prefix = '000'
            if args.column == 1:    # 1 = "% Aciertos"
                prefix = "{:.4f}".format(stats[idx, 1])
            elif args.column == 2:  # 2 = "Mean (Diff)"
                prefix = "{:.10f}".format(stats[idx, 2])
            elif args.column == 3:  # 3 = "DevStd (Diff)"
                prefix = "{:.16f}".format(stats[idx, 3])

            new_name = prefix + '_' + old_name         # 0.0998234_image00123.bmp

            # stats[idx, 0] == "class"
            if int(stats[idx, 0]) == 0:
                path = os.path.join(args.output_dir, 'no_neoplasico')
                destination = os.path.join(path , new_name)
            elif int(stats[idx, 0]) == 1:
                path = os.path.join(args.output_dir, 'neoplasico')
                destination = os.path.join(path, new_name)
            else:
                print("Error: unknown class number")
                sys.exit()

            print("destination: {}".format(destination))

            if not os.path.exists(path):
                os.makedirs(path)

            # copy a file from source to the destination with a new name.
            shutil.copy(source, destination)


    # Example:
    # python save_prefix_image.py -i "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox" -d "data.csv" -o "/home/willytell/Experiments/feed_cnn" -a "feed_cnn"
    #
    # notebook: 
    # python save_prefix_image.py -i "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox" -d "data.csv" -o "/home/willytell/Experiments/feed_cnn" -a "feed_cnn"
    elif args.action == 'feed_cnn':
        # "Image Name", "class", "% Aciertos", "Mean (Diff)", "DevStd (Diff)"
        X = np.genfromtxt(args.data, delimiter=",", usecols=(0), dtype='str')
        y = np.genfromtxt(args.data, delimiter=",", usecols=(1), dtype=int)

        # We have only 2 classes
        # noneo = class 0
        # neo   = class 1

        X_noneo, y_noneo = filter_by_class(X, y, 0)
        X_neop,  y_neop  = filter_by_class(X, y, 1)


        #experiment0_dataset_24_5_X_test_neop.npy
        #experiment0_dataset_24_5_X_test_noneo.npy
        #experiment0_dataset_24_5_y_test_neop.npy
        #experiment0_dataset_24_5_y_test_noneo.npy
        cad = 'experiment0_dataset_24_5'

        # Saving the X_test and y_test for Neop and Noneo
        np.save(os.path.join(args.output_dir, cad + '_X_test_neop'), X_neop)
        np.save(os.path.join(args.output_dir, cad + '_y_test_neop'), y_neop)

        np.save(os.path.join(args.output_dir, cad + '_X_test_noneo'), X_noneo)
        np.save(os.path.join(args.output_dir, cad + '_y_test_noneo'), y_noneo)


    # Example:
    # Option -c: -c 1 = "% Aciertos", -c 2 = "Mean (Diff)", -c 3 = "DevStd (Diff)"
    # python save_prefix_image.py -d "ranking.csv" -c 1 -t 0.8 -a "selection" > selection_from_ranking_0.8.csv
    elif args.action == 'selection':
        #"Image Name", "class", "% Aciertos", "Mean (Diff)", "DevStd (Diff)"
        image_names = np.genfromtxt(args.data, delimiter=",", usecols=(0,1,2), dtype='str')
        stats = np.genfromtxt(args.data, delimiter=",", usecols=(3,4,5,6))

        # We have only 2 classes
        # noneo = class 0
        # neo   = class 1


        # Option -c: -c 1 = "% Aciertos", -c 2 = "Mean (Diff)", -c 3 = "DevStd (Diff)"
        for idx, fname in enumerate(image_names):
            if args.column == 1:  # 1 = "% Aciertos"
                if stats[idx, 1] >= args.threshold:
                    print("{};{};{};{};{};{};{}".format(image_names[idx,0],
                                                     image_names[idx,1],
                                                     image_names[idx,2],
                                                     stats[idx, 0],
                                                     stats[idx, 1],
                                                     stats[idx, 2],
                                                     stats[idx, 3]))
            else:
                print("Unknown column.")



    # Example:
    # python save_prefix_image.py -i "/home/willytell/polcol" -im "images_filenames_from_ranking_0.8.txt" -l "binary_classification_from_ranking_0.8.txt"
    #                             -config "/home/willytell/polcol/config/conf_dist50-resnet50-bbox-exp0.py" -e 0 -k 0 -o "test-0.8"

    elif args.action == 'test':
        # make a "test set" including all the images in the data set, less the images that are in the train and validation set.

        config_path = args.configuration #"/home/willytell/polcol/config/conf_dist50-resnet50-bbox-exp0.py"

        cf = SourceFileLoader('config', config_path).load_module()

        # read the dataset
        X, y = read_data(args.input_dir, args.images, args.labels)

        if cf.num_classes == 2:
            X_test_noneo, y_test_noneo = filter_by_class(X, y, 'NONEOPLASICO')
            X_test_neop, y_test_neop = filter_by_class(X, y, 'NEOPLASICO')


        cf.output_path = os.path.join(cf.experiments_path, cf.experiment_name, args.output_dir)

        if not os.path.exists(cf.output_path):
            os.makedirs(cf.output_path)

        #experiment9_dataset_21_9_X_test_neop.npy
        #experiment9_dataset_21_9_y_test_neop.npy

        #experiment9_dataset_21_9_X_test_noneo.npy
        #experiment9_dataset_21_9_y_test_noneo.npy

        e = args.experiment
        k = args.kfold
        data_path = os.path.join(cf.experiments_path, cf.experiment_name) + '/' + cf.experiment_prefix + str(
            e) + '_' + cf.dataset_prefix + '_' + str(cf.num_images_for_test) + '_' + str(
            cf.n_splits) + '_' + cf.n_splits_prefix + str(k)


        modes = ["train", "validation"]

        for mode in modes:
            print("mode: {}".format(mode))
            # Load Neop: X_train/validation/test_neop.npy, y_train/validation/test_neop.npy
            X_neop = np.load(data_path + '_X_' + mode + '_neop.npy')
            y_neop = np.load(data_path + '_y_' + mode + '_neop.npy')
            print("\nReading X_neop:")
            print(X_neop)
            print(y_neop)

            # Load NOneo: X_train/validation/test_noneo.npy, y_train/validation/test_noneo.npy
            X_noneo = np.load(data_path + '_X_' + mode + '_noneo.npy')
            y_noneo = np.load(data_path + '_y_' + mode + '_noneo.npy')
            print("\nReading X_noneo:")
            print(X_neop)
            print(y_neop)

            # Neop class
            for item in X_neop:
                idx = np.where(X_test_neop == item)
                if np.asarray(idx).size == 0:
                    print("X_neop: idx is zero.")
                X_test_neop = np.delete(X_test_neop, idx, axis=0)
                y_test_neop = np.delete(y_test_neop, idx, axis=0)

            # No-neo class
            for item in X_noneo:
                idx = np.where(X_test_noneo == item)
                if np.asarray(idx).size == 0:
                    print("X_noneo: idx is zero.")
                X_test_noneo = np.delete(X_test_noneo, idx, axis=0)
                y_test_noneo = np.delete(y_test_noneo, idx, axis=0)


        cad = cf.experiment_prefix + str(e) + '_' + cf.dataset_prefix + '_' + str(cf.num_images_for_test) + '_' + str(
            cf.n_splits)

        print("cad: {}".format(cad))

        # Saving the X_test and y_test for Neop and Noneo
        # np.save(os.path.join(cf.output_path, cad + '_X_test_neop'), X_test_neop)
        # np.save(os.path.join(cf.output_path, cad + '_y_test_neop'), y_test_neop)

        # np.save(os.path.join(cf.output_path, cad + '_X_test_noneo'), X_test_noneo)
        # np.save(os.path.join(cf.output_path, cad + '_y_test_noneo'), y_test_noneo)

        print("\nWriting X_test_neop:")
        print(X_test_neop)
        print(y_test_neop)

        print("\nWriting X_test_noneo:")
        print(X_test_noneo)
        print(y_test_noneo)

    
    else:
        print("Unknown action.")


















