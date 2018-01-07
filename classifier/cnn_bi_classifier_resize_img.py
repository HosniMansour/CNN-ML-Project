from PIL import Image
import os
from os import path

train_data_dir = 'bi_classifier_data/training'
validation_data_dir = 'bi_classifier_data/validation'
test_data_dir = 'bi_classifier_data/testing'

def resize(file):
    print('resizing ' + file)
    img = Image.open(file)
    img = img.resize((150, 150), Image.ANTIALIAS)
    img.save(file)

def resizeAllInFolder(folder):
    for dir in os.listdir(folder):
        dir = train_data_dir + '/' + dir
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                file = dir + '/' + file
                if os.path.isfile(file):
                    resize(file)


resizeAllInFolder(train_data_dir)
resizeAllInFolder(validation_data_dir)
resizeAllInFolder(test_data_dir)
