from keras.preprocessing.image import load_img, img_to_array

img = load_img('bi_classifier_data/training/cat/cat.0.jpg')
x = img_to_array(img)
print(x.shape)