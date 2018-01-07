from keras.models import model_from_json
import os
from PIL import Image
import numpy as np
import cv2

class FacialClassifier(object):
    facial_model = None
    map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
                      3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
                      7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
                      11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
                      14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

    pic_size = 64

    def __init__(self):
        # load and configure the binary classifier model for "cats vs dogs"
        self.facial_model = model_from_json(
            open(os.path.join('../classifier/models/simpson', 'model.json')).read())
        self.facial_model.load_weights(os.path.join('../classifier/models/simpson', 'weights_6conv.hdf5'))
        self.facial_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def run_test(self):
        print(self.predict('../classifier/simpson/SimpsonDataset/bart_simpson/pic_0000.jpg'))

    def predict(self, filename):

        img = cv2.imread(filename)
        img = cv2.resize(img, (self.pic_size, self.pic_size))

        input = np.asarray(img)
        input = input.astype('float32') / 255
        input = np.expand_dims(input, axis=0)

        print(input.shape)

        output = self.facial_model.predict(input)

        for i in range(len(self.map_characters)):
            if (output[0][i] > 0.4):
                predicted_label = self.map_characters[i]
                predicted_char = output[0][i]
                print(predicted_char)
                return predicted_char, predicted_label