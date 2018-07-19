import numpy as np
import os
import cv2
import sys

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


class Crop():
    def __init__(self, width=224, high=224):
        self.suffix = '_Polyp.tif'
        self.connectivity = 4
        self.width = width
        self.high = high

    def load(self, cf):
        self.X_original, self.y_original = read_data(cf.dataset_directory, cf.images_filenames, cf.labels)

        # classes = np.unique(y_all) that is equal to ['NONEO', 'NEO']
        self.classes, self.frequency = np.unique(self.y_original, return_counts=True) # count the frequency of unique values

        self.n_samples = len(self.X_original)
        self.n_classes = len(self.classes)


    def centered_crop(self, image, thresh):

        #output = cv2.connectedComponentsWithStats(thresh, self.connectivity, cv2.CV_32S)
        #nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, self.connectivity, cv2.CV_32S)

        # print("output[2] = ", output[2])
        # print("\n")


        # in general, almost allways thera two elements.
        idx_global = 1

        # but, if there are more than 2, we have to find who has the major amount.
        if stats.shape[0] > 2:
            for idx_tmp in range(1, stats.shape[0]):
                if stats[idx_tmp, 4] > stats[idx_global, 4]:
                    idx_global = idx_tmp

        # output[2]
        #      x      y     w     h    amount px
        # 0 [  0      0   1920  1080 1816041]    <=== corresponds to the whole image
        # 1 [313    121    660   930  257558]
        # 2 [357    870     1     1        1]

        # bounding box
        x = stats[idx_global, 0]
        y = stats[idx_global, 1]
        w = stats[idx_global, 2]
        h = stats[idx_global, 3]

        # centroid
        x_center = centroids[idx_global, 0]
        y_center = centroids[idx_global, 1]
        #print("centroid({}, {})".format(int(x_center), int(y_center)))


        # top left corner of the crop
        x1 = int(x_center) - (self.width // 2)
        y1 = int(y_center) - (self.high  // 2)

        if 0 <= x1 and 0 <= y1 and (x1 + self.width) <= stats[0, 2] and (y1 + self.high) <= stats[0, 4]:
            #print(">> crop inside the image!!!")

            crop_thesh = thresh[y1:y1 + self.high, x1:x1 + self.width]
            crop_image = image [y1:y1 + self.high, x1:x1 + self.width]

            # erosion with SE = (width, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.width, self.high))
            #kernel = np.ones((self.width, self.high), np.uint8)
            erosion = cv2.erode(crop_thesh, kernel, iterations=1)

            nlabels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(erosion, self.connectivity, cv2.CV_32S)
            #print(stats.shape)
            #sys.stdout.flush()
            #print(stats2)
            #sys.stdout.flush()

            if stats2.shape[0] == 2 and stats2[1, 0] == 0 and stats2[1, 1] == 0 and stats2[1, 2] == self.width and \
                stats2[1, 3] == self.high and stats2[1, 4] == (self.width * self.high):
                #return True, crop_thesh[0:self.high, 0:self.width]*8
                return True, crop_image[0:self.high, 0:self.width] , crop_thesh[0:self.high, 0:self.width]

            #return True, image[y1:y1+self.high, x1:x1+self.width]

        return False, image[y:y + h, x:x + w], thresh[y:y + h, x:x + w]


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

            print("Processing image: {}".format(os.path.join(cf.dataset_images_path, name[0] + name[1])))


            image_copy = image.copy()
            thresh_copy = thresh.copy()

            count = 2
            stop = False
            while not stop and count < 5:
                flag, roi, thr = self.centered_crop(image, thresh)

                if flag:
                    print("Saving image: {}".format(name[0]))
                    cv2.imwrite(os.path.join(cf.bbox_output_path, name[0] + name[1]), roi)
                    #cv2.imwrite(os.path.join(cf.bbox_output_path, name[0] + '_BIN_'+ name[1]), thr*8)
                    stop = True

                else:
                    #print("skipped crop.")
                    thresh = cv2.resize(thr, (thr.shape[1] * count, thr.shape[0] * count), interpolation=cv2.INTER_CUBIC)
                    image = cv2.resize(roi, (roi.shape[1] * count, roi.shape[0] * count), interpolation=cv2.INTER_CUBIC)
                    count += 1

            if count >= 5:
                print("Too much zoom. Image discarded: {}{}".format(name[0], name[1]))


        print("All images processed.")
