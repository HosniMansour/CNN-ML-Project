from keras.models import model_from_json
import os
from PIL import Image
import numpy as np


class BiClassifier(object):
    bi_model = None

    def __init__(self):
        # load and configure the binary classifier model for "cats vs dogs"
        self.bi_model = model_from_json(
            open(os.path.join('../classifier/models', 'cnn_bi_classifier_architecture.json')).read())
        self.bi_model.load_weights(os.path.join('../classifier/models', 'cnn_bi_classifier_weights.h5'))
        self.bi_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def run_test(self):
        print(self.predict('../classifier/bi_classifier_data/training/cat/cat.2.jpg'))

    def predict(self, filename):
        img = Image.open(filename)
        img = img.resize((150, 150), Image.ANTIALIAS)

        input = np.asarray(img)
        input = input.astype('float32') / 255
        input = np.expand_dims(input, axis=0)

        print(input.shape)

        output = self.bi_model.predict(input)

        probability_of_a_dog = output[0][0]
        predicted_label = ("Dog" if probability_of_a_dog > 0.5 else "Cat")
        return probability_of_a_dog, predicted_label
