import numpy as np
import argparse
import os
import sys
import shutil


def filter_by_class(X, y, filter):
    idx = np.where(y == filter)  # filter: 0 = 'NONEOPLASICO' and 1 = 'NEOPLASICO'
    idx_arr = np.asarray(idx[0])  # tupla to np.array

    return np.copy(X[idx_arr]), np.copy(y[idx_arr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save prefix to image')
    parser.add_argument('-i', '--input_dir', type=str, default=None, help='directory with images')
    parser.add_argument('-d', '--data', type=str, default='data.csv', help='csv file with data')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='directory to save images')
    parser.add_argument('-c', '--column_prefix', type=int, default=None, help='column to set the prefix')
    parser.add_argument('-a', '--action', type=str, default=None, help='')
    args = parser.parse_args()

    # Example:
    # Option -c: -c 1 = "% Aciertos", -c 2 = "Mean (Diff)", -c 3 = "DevStd (Diff)"
    # python save_prefix_image.py -i "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox" -d "data.csv" -c 1 -o "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/test"
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
            if args.column_prefix == 1:    # 1 = "% Aciertos"
                prefix = "{:.4f}".format(stats[idx, 1])
            elif args.column_prefix == 2:  # 2 = "Mean (Diff)"
                prefix = "{:.10f}".format(stats[idx, 2])
            elif args.column_prefix == 3:  # 3 = "DevStd (Diff)"
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



