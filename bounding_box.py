import numpy as np
import os
import cv2

from utils import read_data

class BBox():
    def __init__(self):
        self.suffix = '_Polyp.tif'
        self.connectivity = 4

    def load(self, cf):
        self.X_original, self.y_original = read_data(cf.dataset_directory, cf.images_filenames, cf.labels)

        # classes = np.unique(y_all) that is equal to ['NONEO', 'NEO']
        self.classes, self.frequency = np.unique(self.y_original, return_counts=True) # count the frequency of unique values

        self.n_samples = len(self.X_original)
        self.n_classes = len(self.classes)


    def make(self, cf):

        if not os.path.exists(cf.bbox_output_path):
            os.makedirs(cf.bbox_output_path)

        images_stats = []
        files_list = []

        [files_list.append(os.path.splitext(name)) for name in self.X_original]

        for name in files_list:

            #print(name[0])

            image = cv2.imread(os.path.join(cf.dataset_images_path, name[0] + name[1]), flags=cv2.IMREAD_COLOR)

            mask = cv2.imread(os.path.join(cf.dataset_mask_directory, name[0] + self.suffix), cv2.IMREAD_GRAYSCALE)
            ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            stats=np.zeros([], dtype="uint8")
            centroid=np.array([])

            output = cv2.connectedComponentsWithStats(thresh, self.connectivity, cv2.CV_32S)

            #print("output[2] = ", output[2])
            #print("\n")


            # in general, almost allways thera two elements.
            idx_global = 1

            # but, if there are more than 2, we have to find who has the major amount.
            if output[2].shape[0] > 2:

                for idx_tmp in range(1, output[2].shape[0]):
                    if output[2][idx_tmp, 4] > output[2][idx_global, 4]:
                        idx_global = idx_tmp

            # output[2]
            #      x      y     w     h    amount px
            # 0 [  0      0   1920  1080 1816041]    <=== corresponds to the whole image
            # 1 [313    121    660   930  257558]
            # 2 [357    870     1     1        1]

            x = output[2][idx_global, 0]
            y = output[2][idx_global, 1]
            w = output[2][idx_global, 2]
            h = output[2][idx_global, 3]

            roi = image[y:y + h, x:x + w]
            print("Saving image: ", name[0])
            cv2.imwrite(os.path.join(cf.bbox_output_path, name[0] + name[1]), roi)

            images_stats.append(output[2][idx_global])

        # Save information of each image using idx_global (to be able to compute the mean size, max/min size.)
        np.save(os.path.join(cf.bbox_output_path, "images_stats"), np.array(images_stats))