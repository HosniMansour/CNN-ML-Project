import os
from cnn_bi_classifier import BiClassifier
from cnn_facial_classifier import FacialClassifier
from vgg16_classifier import VGG16Classifier

from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

bi_classifier = BiClassifier()
facial_classifier = FacialClassifier()
vgg16_classifier = VGG16Classifier()


@app.route('/')
def classifiers():
    return render_template('classifiers.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def store_uploaded_image(action):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for(action,
                                filename=filename))


@app.route('/cats_vs_dogs', methods=['GET', 'POST'])
def cats_vs_dogs():
    if request.method == 'POST':
        return store_uploaded_image('cats_vs_dogs_result')
    return render_template('cats_vs_dogs.html')


@app.route('/facial', methods=['GET', 'POST'])
def facial():
    if request.method == 'POST':
        return store_uploaded_image('facial_result')
    return render_template('facial.html')


@app.route('/vgg16', methods=['GET', 'POST'])
def vgg16():
    if request.method == 'POST':
        return store_uploaded_image('vgg16_result')
    return render_template('vgg16.html')


@app.route('/cats_vs_dogs_result/<filename>')
def cats_vs_dogs_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    probability_of_dog, predicted_label = bi_classifier.predict(filepath)
    return render_template('cats_vs_dogs_result.html', filename=filename,
                           probability_of_dog=probability_of_dog, predicted_label=predicted_label)


@app.route('/facial_result/<filename>')
def facial_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Get this from new model --> it Work now hah :D
    predicted_char, predicted_label = facial_classifier.predict(filepath)
    return render_template('facial_result.html', filename=filename,
                           predicted_char=predicted_char, predicted_label=predicted_label)

@app.route('/vgg16_result/<filename>')
def vgg16_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    top3 = vgg16_classifier.predict(filepath)
    return render_template('vgg16_result.html', filename=filename,
                           top3=top3)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def main():
    bi_classifier.run_test()
    facial_classifier.run_test()
    vgg16_classifier.run_test()
    app.run(debug=True)


if __name__ == '__main__':
    main()
