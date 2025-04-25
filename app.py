from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
model = load_model('brain_tumor_model.h5')  # Make sure this file is in your project directory

# Set the input image size your model expects
IMAGE_SIZE = (128, 128)  # <- Changed from 224 to 128

# Home Route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            img = image.load_img(filepath, color_mode='grayscale', target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 128, 128, 1)
            img_array = img_array / 255.0  # Normalize

            # Predict using the model
            result = model.predict(img_array)
            confidence = float(np.max(result)) * 100  # For softmax output

            # Interpret result
            class_index = np.argmax(result)
            if class_index == 1:
                prediction = "Brain Tumor detected"
            else:
                prediction = "No Brain Tumor detected"

            # Path for displaying image in HTML
            image_path = f'uploads/{filename}'

    return render_template(
        'index2.html',
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
