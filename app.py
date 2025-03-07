from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('InceptionV3_model.h5')

# Ensure the 'static/uploads' folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def upload():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Process the image
    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    class_labels = ['Bison', 'Elephant', 'Horse', 'Lion', 'Tiger']
    result = class_labels[predicted_class]

    return render_template('after.html', prediction=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)

