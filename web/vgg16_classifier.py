from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np


class VGG16Classifier:
    model = None

    def __init__(self):
        self.model = VGG16(include_top=True, weights='imagenet')
        self.model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, filename):
        img = Image.open(filename)
        img = img.resize((224, 224), Image.ANTIALIAS)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        output = decode_predictions(self.model.predict(input), top=3)
        return output[0]

    def run_test(self):
        print(self.predict('../classifier/bi_classifier_data/training/cat/cat.3.jpg'))


def main():
    classifier = VGG16Classifier()
    classifier.run_test()

if __name__ == '__main__':
    main()
