import re
import chardet

def parser(text):
    enc = chardet.detect(open(text, "rb").read())['encoding']
    fp_r = open(text, "r",encoding = enc)
    label_set = set()

    for aLine in fp_r:
        string = re.sub('[^0-9]', ' ', aLine)
        label_set.update(int(x) for x in string.split())
    fp_r.close()
    return sorted(list(label_set), reverse=True)

def label_checker(label_to_check, label_from_text, dest ):
    fp_r = open(label_to_check, "r")
    fp_w = open(dest, "w")
    label_coord = []
    for aLine in fp_r:
        label_coord.append(aLine.replace('\n', '').split(','))
    sorted_label = sorted(label_coord, key=lambda raw_coord: int(raw_coord[0]), reverse=True)

    #1. remove exactly same label set with probabilty over 90%
    to_pop_label = []
    to_pop_text = []
    for i in range(0, len(sorted_label)):
        for j in range(0, len(label_from_text)):
            if sorted_label[i][0] == str(label_from_text[j]) and float(sorted_label[i][1]) >= 0.9:
                to_pop_label.append(i)
                to_pop_text.append(j)
                print(", ".join(sorted_label[i]), file=fp_w)
                break
    indexes = sorted(to_pop_label, reverse=True)
    for index in indexes:
        del sorted_label[index]
    indexes = sorted(to_pop_text, reverse=True)
    for index in indexes:
        del label_from_text[index]

    # 2. remove exactly same label set with probabilty over 50%
    to_pop_label = []
    to_pop_text = []
    for i in range(0, len(sorted_label)):
        for j in range(0, len(label_from_text)):
            if sorted_label[i][0] == str(label_from_text[j]) and float(sorted_label[i][1]) >= 0.7:
                to_pop_label.append(i)
                to_pop_text.append(j)
                print(", ".join(sorted_label[i]), file=fp_w)
                break
    indexes = sorted(to_pop_label, reverse=True)
    for index in indexes:
        del sorted_label[index]
    indexes = sorted(to_pop_text, reverse=True)
    for index in indexes:
        del label_from_text[index]

    #3. match if both have same length and pass label_cmp test
    to_pop_label = []
    to_pop_text = []
    for i in range(0, len(sorted_label)):
        for j in range(0, len(label_from_text)):
            if len(sorted_label[i][0]) == len(str(label_from_text[j])) \
                    and label_cmp(sorted_label[i][0], str(label_from_text[j])):
                to_pop_label.append(i)
                to_pop_text.append(j)
                del(sorted_label[i][0])
                print(str(label_from_text[j]) + ", " + ", ".join(sorted_label[i]), file=fp_w)
                break
    indexes = sorted(to_pop_label, reverse=True)
    for index in indexes:
        del sorted_label[index]
    indexes = sorted(to_pop_text, reverse=True)
    for index in indexes:
        del label_from_text[index]


def label_cmp(img_label, text_label):
    score = 0
    for i in range(0, len(img_label)):
        if img_label[i] == text_label[i]:
            score += 1
        elif img_label[i] == '7' and text_label[i] == '1':
            score += 1
        elif img_label[i] == '1' and text_label[i] == '7':
            score += 1
        elif img_label[i] == '6' and text_label[i] == '8':
            score += 1
        elif img_label[i] == '6' and text_label[i] == '5':
            score += 1
        elif img_label[i] == '5' and text_label[i] == '6':
            score += 1
        elif img_label[i] == '5' and text_label[i] == '8':
            score += 1
        elif img_label[i] == '8' and text_label[i] == '5':
            score += 1
        elif img_label[i] == '8' and text_label[i] == '6':
            score += 1

    if score == len(img_label):
        return True
    else:
        return False
