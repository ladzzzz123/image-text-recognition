from image_processor import ImageProcess
import find_labels as fl
import label_generator as lg
if __name__ == '__main__':
    img_name = input("name: ")
    processor = ImageProcess('images/'+img_name)
    # plots preprocessed image
    processor.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = processor.get_candidates()
    # plots objects detected
    processor.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    text = processor.predict_char()
    # plots the realigned text
    raw_result = processor.realign_text()
    fl.union_and_find('labels/'+img_name.split('.')[0]+'_single.txt', raw_result)
    lg.char_concatenate('labels/'+img_name.split('.')[0]+'_single.txt', 'labels/'+img_name.split('.')[0]+'_label.txt')
    lg.plot_labels('labels/'+img_name.split('.')[0]+'_label.txt', 'images/'+img_name)


