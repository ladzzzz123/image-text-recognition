from image_processor import ImageProcess

if __name__ == '__main__':
    img_name = input("name: ")
    processor = ImageProcess('images\\'+img_name)
    # plots preprocessed image
    processor.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = processor.get_candidates()
    # plots objects detected
    processor.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    text = processor.predict_char()
    # plots objects after text detection
    processor.plot_to_check(text, 'Text Detected from Objects')
    # plots the realigned text
    processor.realign_text()