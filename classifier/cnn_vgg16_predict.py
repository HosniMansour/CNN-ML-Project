from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np

model = VGG16(include_top=True, weights='imagenet')
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])


def predict(filename):
    img = Image.open(filename)
    img = img.resize((224, 224), Image.ANTIALIAS)
    input = img_to_array(img)
    input = np.expand_dims(input, axis=0)
    input = preprocess_input(input)
    output = decode_predictions(model.predict(input), top=3)
    print(output)

# 7asesha zayda hathi :v
#for i in range(100):
#    predict('bi_classifier_data/training/cat/cat.' + str(i) + '.jpg')
#    predict('bi_classifier_data/training/dog/dog.' + str(i) + '.jpg')
