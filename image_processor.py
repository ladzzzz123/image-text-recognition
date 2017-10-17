import numpy as np
import cv2

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize

from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage import restoration
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches

import cnn_predictor as cnn
#import cnn_chars74k as cnn


class ImageProcess():
    def __init__(self, image_file):
        tmp_img = cv2.imread(image_file, 0)
        #tmp_img = cv2.medianBlur(tmp_img, 5)
        #tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 0)
        tmp_img = cv2.bitwise_not(tmp_img)
        self.image = np.asarray(tmp_img)
        #self.image = imread(image_file, as_grey=True)
        #self.image = 1-imread(image_file, as_grey=True)
        self.preprocess()

    def preprocess(self):
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared

    def get_candidates(self):
        label_image = measure.label(self.cleared)
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1
        coordinates = []
        i = 0

        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr - margin, minc - margin, maxr + margin, maxc + margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0] * roi.shape[1] == 0:
                    continue
                else:
                    if i == 0:
                        samples = resize(roi, (28, 28), mode='constant')
                        #samples = resize(roi, (22, 30), mode='constant')
                        coordinates.append(region.bbox)
                        i += 1
                    elif i == 1:
                        roismall = resize(roi, (28, 28), mode='constant')
                        #roismall = resize(roi, (22, 30), mode='constant')
                        samples = np.concatenate((samples[None, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)
                        i += 1
                    else:
                        roismall = resize(roi, (28, 28), mode='constant')
                        #roismall = resize(roi, (22, 30), mode='constant')
                        samples = np.concatenate((samples[:, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)

        self.candidates = {
            'fullscale': samples,
            'flattened': samples.reshape((samples.shape[0], -1)),
            'coordinates': np.array(coordinates)
        }

        print('Images After Contour Detection')
        print('Fullscale: ', self.candidates['fullscale'].shape)
        print('Flattened: ', self.candidates['flattened'].shape)
        print('Contour Coordinates: ', self.candidates['coordinates'].shape)
        print('============================================================')

        return self.candidates

    def plot_preprocessed_image(self):
        """
        plots pre-processed image. The plotted image is the same as obtained at the end
        of the get_text_candidates method.
        """
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = bw.copy()

        label_image = measure.label(cleared)
        borders = np.logical_xor(bw, cleared)

        label_image[borders] = -1
        image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image):
            if region.area < 10:
                continue

            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        plt.show()

    def plot_to_check(self, what_to_plot, title):
        """
        plots images at several steps of the whole pipeline, just to check output.
        what_to_plot is the name of the dictionary to be plotted
        """
        n_images = what_to_plot['fullscale'].shape[0]

        fig = plt.figure(figsize=(12, 12))

        if n_images <= 100:
            if n_images < 100:
                total = range(n_images)
            elif n_images == 100:
                total = range(100)

            for i in total:
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][i], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][i]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()
        else:
            total = list(np.random.choice(n_images, 100))
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][j], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][j]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()

    def predict_char(self):
        """
        it takes as argument a pickle model_mnist and predicts whether the detected objects
        contain text or not.
        """
        predicted = cnn.char_prediction(self.candidates['flattened'])
        self.which_text = {
                                 'fullscale': self.candidates['fullscale'],
                                 'flattened': self.candidates['flattened'],
                                 'coordinates': self.candidates['coordinates'],
                                 'predicted_char': predicted
                                 }
        return self.which_text

    def realign_text(self):
        """
        processes the classified characters and reorders them in a 2D space
        generating a matplotlib image.
        """
        max_maxrow = max(self.which_text['coordinates'][:, 2])
        min_mincol = min(self.which_text['coordinates'][:, 1])
        subtract_max = np.array([max_maxrow, min_mincol, max_maxrow, min_mincol])
        flip_coord = np.array([-1, 1, -1, 1])

        coordinates = (self.which_text['coordinates'] - subtract_max) * flip_coord

        ymax = max(coordinates[:, 0])
        xmax = max(coordinates[:, 3])

        predicted = self.which_text['predicted_char']
        coordinates = [list(coordinate) for coordinate in coordinates]

        #solves python3 zip() problem
        realign_tmp = list(zip(coordinates, predicted))
        to_realign_tmp = realign_tmp[:]
        to_realign = list(to_realign_tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for char in to_realign:
            if (char[1] != 11):
                ax.text(char[0][1], char[0][2], char[1], size=16)
        ax.set_ylim(-10, ymax + 10)
        ax.set_xlim(-10, xmax + 10)

        plt.show()
