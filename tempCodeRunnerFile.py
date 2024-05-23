from flask import Flask, flash, request, redirect, url_for, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import numpy as np
from wtforms.validators import InputRequired
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = '/Users/apple/picture_upload_website/picture22'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()  # Create an instance of the form
    if request.method == 'POST':
        # Check if a file was included in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        # Check if the file is allowed (e.g., image file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load the saved model
            model = load_model('/Users/apple/Desktop/facialemotionmodel.h5')

            # Load and preprocess the uploaded image
            img = image.load_img(file_path, target_size=(48, 48), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make a prediction
            prediction = model.predict(img_array)

            # Define emotion labels
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

            # Get the predicted emotion label
            predicted_label = np.argmax(prediction)
            predicted_emotion = emotion_labels[predicted_label]

            # Pass the predicted emotion to the template
            return render_template('index.html', form=form, filename=filename, predicted_emotion=predicted_emotion)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
