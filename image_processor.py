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
import tensorflow as tf
#import cnn_custom_predictor as cnn
#import cnn_predictor_modified as cnn


class ImageProcess():
    def __init__(self, image_file):
        tmp_img = cv2.imread(image_file, 0)
        #tmp_img = cv2.medianBlur(tmp_img, 5)
        #tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 0)
        tmp_img = cv2.bitwise_not(tmp_img)
        self.image = np.asarray(tmp_img)
        self.preprocess()

    def preprocess(self):
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared

    def get_candidates(self):
        """
        identifies objects in the image. Gets contours, draws rectangles around them
        and saves the rectangles as individual images.
        """
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
                        #resizing individual image into 14 x 20 pixel image
                        #This is because usual font images have longer height than width
                        samples = resize(roi, (20, 14), mode='constant')
                        #padding extra pixel to zero in order to fit it in 28x28 convolution layer
                        samples = np.pad(samples, ((4, 4), (7, 7)),  mode='constant')
                        coordinates.append(region.bbox)
                        i += 1
                    elif i == 1:
                        roismall = resize(roi, (20, 14), mode='constant')
                        roismall = np.pad(roismall, ((4, 4), (7, 7)), mode='constant')
                        samples = np.concatenate((samples[None, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)
                        i += 1
                    else:
                        roismall = resize(roi, (20, 14), mode='constant')
                        roismall = np.pad(roismall, ((4, 4), (7, 7)), mode='constant')
                        samples = np.concatenate((samples[:, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)

        self.candidates = {
            'fullscale': samples,
            'flattened': samples.reshape((samples.shape[0], -1)),
            'coordinates': np.array(coordinates)
        }

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
        plt.show(block=False)


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
                    ax.text(-6, 8, "{}".format(str(what_to_plot['predicted_char'][i])), fontsize=22, color='red')
                    #ax.text(-6, 8, "{}: {:.2}".format(str(what_to_plot['predicted_char'][i]),
                            #float(max(what_to_plot['predicted_prob'][i].tolist()))), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show(block=False)

        else:
            total = list(np.random.choice(n_images, 100))
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][j], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, "{}".format(str(what_to_plot['predicted_char'][i])), fontsize=22, color='red')
                    #ax.text(-6, 8, "{}: {:.2}".format(str(what_to_plot['predicted_char'][i]),
                            #float(max(what_to_plot['predicted_prob'][i].tolist()))), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show(block=False)


    def predict_char(self):
        """
        it takes as argument a pickle model_mnist and predicts whether the detected objects
        contain text or not.
        """
        argmax = []
        softmax = []
        prob = []
        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        g4 = tf.Graph()
        g5 = tf.Graph()

        with g1.as_default():
            argmax_tmp, softmax_tmp = cnn.char_prediction(self.candidates['flattened'], 1)
            prob_tmp = [max(list(softmax_tmp)) for softmax_tmp in softmax_tmp]
            argmax.append(argmax_tmp)
            softmax.append(softmax_tmp)
            prob.append(prob_tmp)

        with g2.as_default():
            argmax_tmp, softmax_tmp = cnn.char_prediction(self.candidates['flattened'], 2)
            prob_tmp = [max(list(softmax_tmp)) for softmax_tmp in softmax_tmp]
            argmax.append(argmax_tmp)
            softmax.append(softmax_tmp)
            prob.append(prob_tmp)

        with g3.as_default():
            argmax_tmp, softmax_tmp = cnn.char_prediction(self.candidates['flattened'], 3)
            prob_tmp = [max(list(softmax_tmp)) for softmax_tmp in softmax_tmp]
            argmax.append(argmax_tmp)
            softmax.append(softmax_tmp)
            prob.append(prob_tmp)

        with g4.as_default():
            argmax_tmp, softmax_tmp= cnn.char_prediction(self.candidates['flattened'], 4)
            prob_tmp = [max(list(softmax_tmp)) for softmax_tmp in softmax_tmp]
            argmax.append(argmax_tmp)
            softmax.append(softmax_tmp)
            prob.append(prob_tmp)

        with g5.as_default():
            argmax_tmp, softmax_tmp = cnn.char_prediction(self.candidates['flattened'], 5)
            prob_tmp = [max(list(softmax_tmp)) for softmax_tmp in softmax_tmp]
            argmax.append(argmax_tmp)
            softmax.append(softmax_tmp)
            prob.append(prob_tmp)

        predicted_argmax = argmax[0]
        predicted_softmax = softmax[0]
        predicted_prob = prob[0]
        for i in range(1, 5):
            if predicted_prob < prob[i]:
                predicted_argmax = argmax[i]
                predicted_softmax = softmax[i]
                predicted_prob = prob[i]

        self.which_text = {
                                 'fullscale': self.candidates['fullscale'],
                                 'flattened': self.candidates['flattened'],
                                 'coordinates': self.candidates['coordinates'],
                                 'predicted_char': predicted_argmax,
                                 'predicted_prob': predicted_softmax
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

        predicted_char = self.which_text['predicted_char']
        predicted_prob = self.which_text['predicted_prob']
        predicted_prob = [max(list(predicted_prob)) for predicted_prob in predicted_prob]
        coordinates = [list(coordinate) for coordinate in coordinates]

        #solves python3 zip() problem
        realign_tmp = list(zip(coordinates, predicted_char, predicted_prob))
        to_realign_tmp = realign_tmp[:]
        to_realign = list(to_realign_tmp)

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        for char in to_realign:
            if(char[2] <0.5):
                pass
                #ax.text(char[0][1], char[0][2], char[1], size=16, color='black')
            elif (char[2] <0.70):
                if(char[1] == 1):
                    #pass
                    ax.text(char[0][1], char[0][2], char[1], size=16, color='black')
                else:
                    pass
                    #ax.text(char[0][1], char[0][2], char[1], size=16, color='black')
            elif(char[2] <0.90):
                if(char[1] == 1 or char[1] == 6):
                    #pass
                    ax.text(char[0][1], char[0][2], char[1], size=16, color='red')
            elif (char[2] < 0.95):
                #pass
                ax.text(char[0][1], char[0][2], char[1], size=16, color='blue')
            else:
                ax.text(char[0][1], char[0][2], char[1], size=16, color='green')

        ax.set_ylim(-10, ymax + 10)
        ax.set_xlim(-10, xmax + 10)

        plt.show()
