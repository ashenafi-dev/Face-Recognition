from flask import Flask, render_template, request
import base64
import numpy as np
# from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load the trained face recognition model
# model = load_model('model/my_model.h5')
    
@app.route('/home', methods=['POST'])
def home():
        # read the image to be recognised
        image_path = request.form.get('imagePath')
        # image = cv.imread(image_path)
        image = Image.open(image_path)
        imageDisplay = Image.open(image_path)
        imageDisplayArray = np.array(imageDisplay)

        # resize the image to match the trained network dimensions
        img_resized = image.resize((64, 64))
        img_data = np.array(img_resized)


        # Preprocess the test image (normalize pixel values)
        test_image = np.expand_dims(img_data, axis=0)
        test_image = test_image / 255.0

        # Make predictions
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions)


        # personal details coresponding to classes
        person_details = {
            'abraham_dessie': {'name': 'Abraham Dessie', 'Department': 'Computer Science', 'personality': 'hard worker'},
            'amare_chanie': {'name': 'Amare Chanie', 'Department': 'Computer Science', 'personality': 'humble'},
            'ashenafi_yirgalem': {'name': 'Ashenafi Yirgalem', 'Department': 'Computer Science', 'personality': 'creative'},
            'habtamu_wale': {'name': 'Habtamu Wale', 'Department': 'Computer Science', 'personality': 'leader'},
            'tadla_eshetie': {'name': 'Tadla Eshetie', 'Department': 'Computer Science', 'personality': 'wholesome'}
        }

        # class labels
        class_labels = ['abraham_dessie', 'amare_chanie', 'ashenafi_yirgalem','habtamu_wale', 'tadla_eshetie']

        # Get the recognized person's details
        predicted_class_name = class_labels[predicted_class_index]
        recognized_person = person_details.get(predicted_class_name, {})
        confidence_score = predictions[0][predicted_class_index] * 100 

        # Display results
        Name = recognized_person.get('name', 'Unknown')
        department = recognized_person.get('Department', 'Unknown')
        personality = recognized_person.get('personality', 'Unknown')

        # Encode the image as base64
        image_64bit = imageDisplayArray.astype(np.uint8)
        buffer = io.BytesIO()  # Create an in-memory buffer
        Image.fromarray(image_64bit).convert('RGB').save(buffer, format='PNG')
        image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # return
        return render_template('home.html', image=image, name=Name, confidance=f"{confidence_score:.2f}%", Department=department,personality=personality);

if __name__ == '__main__':
    app.run()