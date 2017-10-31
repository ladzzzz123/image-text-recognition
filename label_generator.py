from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
import numpy as np


def char_concatenate(source, dest):
    fp_r = open(source, 'r')
    fp_w = open(dest, 'w')
    for aLine in fp_r:
        coord_list = aLine.split(';')
        raw_char = []
        for coord in coord_list:
            c, prob, minx, miny, maxx, maxy = coord.split(',')
            c = int(c); prob = float(prob); minx = int(minx); miny = int(miny); maxx = int(maxx); maxy = int(maxy)
            raw_char.append([c, prob, minx, miny, maxx, maxy])
        sorted_char = sorted(raw_char, key=lambda raw_char: raw_char[2])
        label = ""
        min_prob = 1
        max_prob = 0
        for digit in sorted_char:
            label += str(digit[0])
            if min_prob > digit[1]:
                min_prob = digit[1]
            if max_prob < digit[1]:
                max_prob = digit[1]

        min_x = sorted_char[0][2]
        min_y = sorted_char[0][3]
        max_x = sorted_char[len(sorted_char)-1][4]
        max_y = sorted_char[len(sorted_char)-1][5]

        print("{}, {:.3f}, {:.3f}, {}, {}, {}, {}".format(label, min_prob, max_prob, min_x, min_y, max_x, max_y), file=fp_w)

    fp_r.close()
    fp_w.close()

def plot_labels(source, image_file):
    fp = open(source, 'r')
    label_list=[]
    for aLine in fp:
        label, min_p, max_p, minx, miny, maxx, maxy = aLine.split(',')
        min_p = float(min_p); max_p = float(max_p); minx = int(minx); miny = int(miny); maxx = int(maxx); maxy = int(maxy)
        label_list.append([label, min_p, max_p, minx, miny, maxx, maxy])

    image = imread(image_file, as_grey=True)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(round(image.shape[1]/100), round(image.shape[0]/100)))

    ax.imshow(np.flipud(image))
    for label in label_list:
        if label[2] < 0.6:
            continue
        elif label[2] < 0.9:
            pass
            rect = mpatches.Rectangle((label[3]+10, label[4]+10), label[5] - label[3]+10, label[6] - label[4]+10,
                              fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(label[5]+10, label[6]+10, label[0], size=16, color='red')

        elif label[1] > 0.95:
            rect = mpatches.Rectangle((label[3]+10, label[4]+10), label[5] - label[3]+10, label[6] - label[4]+10,
                              fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(label[5]+10, label[6]+10, label[0], size=16, color='green')
        else:
            pass
            rect = mpatches.Rectangle((label[4]+10, label[3]+10), label[5] - label[3]+10, label[6] - label[4]+10,
                              fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(label[5]+10, label[6]+10, label[0], size=16, color='red')

    ymax = image.shape[0]
    xmax = image.shape[1]
    ax.set_ylim(-10, ymax + 10)
    ax.set_xlim(-10, xmax + 10)
    plt.show()
