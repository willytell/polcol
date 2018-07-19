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

        output = cv2.connectedComponentsWithStats(thresh, self.connectivity, cv2.CV_32S)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage)

        # print("output[2] = ", output[2])
        # print("\n")


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

        # BBox
        roi = image[y:y + h, x:x + w]

        y_center = y + (h // 2)
        x_center = x + (w // 2)

        y1 = y_center - (self.high // 2)
        x1 = x_center - (self.width // 2)

        # check if the crop is inside of the bounding box
        if y1 >= 0 and x1 >= 0 and (y1 + self.high) <= h and (x1 + self.width) <= w:
            print("The centered crop is inside the bbox.")
            crop = thresh[y1:y1 + self.high, x1:x1 + self.width]

            # erosion with SE = (width, high)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.width, self.high))
            kernel = np.ones((self.width, self.high), np.uint8)
            erosion = cv2.erode(crop, kernel, iterations=1)

            output2 = cv2.connectedComponentsWithStats(erosion, self.connectivity, cv2.CV_32S)

            if output2[2].shape[0] > 2:
                amount = output2[2][1, 4]
                if amount == 1:
                    print("Voala!")
                    return True, image[y1:y1 + self.high, x1:x1 + self.width]


        return False, None


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
            #print("".format())

            image_width = image.shape[0]
            image_high = image.shape[1]

            mask_width = mask.shape[0]
            mask_high = mask.shape[1]
            count = 1
            stop= False
            while not stop and count < 100:
                flag, image = self.centered_crop(image, thresh)

                if flag:
                    print("done!")
                    print("Saving image: {}{}".format(name[0], name[1]))
                    cv2.imwrite(os.path.join(cf.bbox_output_path, name[0] + name[1]), image)
                    stop = True

                else:
                    image_width = image_width * count
                    image_high = image_high * count
                    print("image size: {}x{}".format(image_width, image_high))

                    mask_width = mask_width * count
                    mask_high = mask_high * count
                    print("mask size: {}x{}".format(mask_width, mask_high))


                    #thresh = cv2.resize(thresh, (mask_width, mask_high), interpolation=cv2.INTER_CUBIC)
                    #thresh = cv2.resize(thresh, (2000, 2000)) #, interpolation=cv2.INTER_CUBIC)
                    #image  = cv2.resize(image, (image_width, image_high), interpolation=cv2.INTER_CUBIC)
                    #image  = cv2.resize(image, (2000, 2000)) #, interpolation=cv2.INTER_CUBIC)

                    count += 1

        print("All images processed.")
