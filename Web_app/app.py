import os
import io
import string
from turtle import title
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import *
from flask import Flask, request, render_template, redirect, url_for, jsonify
import uuid
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Model
checkpoint_dir = "checkpoint/mymodel-020-0.1008"
model = keras.models.load_model(checkpoint_dir)

class_names = [
				'Bacterial spot', 
				'Early blight', 
				'Healthy', 
				'Late blight', 
				'Leaf mold', 
				'Septoria leaf spot', 
				'Spider mites two-spotted spider mite', 
				'Target spot', 
				'Tomato mosaic virus', 
				'Tomato yellow leaf curl virus']

def transform_image(image_path):
	#img = Image.open(io.BytesIO(image_bytes))
	#img = Image.open(image_path)
	img = load_img(image_path, target_size=(224, 224))
	#target_size = (224, 224)
	#img = img.resize(target_size)
	img_array = img_to_array(img)
	img_array = img_array/255.
	img_array = tf.expand_dims(img_array, 0)

	return img_array

def get_prediction(image_path):
	image = transform_image(image_path)
	predictions = model.predict(image)
	prediction_name = class_names[np.argmax(predictions[0])]
	prediction_confidence = 100 * np.max(predictions[0])

	return prediction_name, prediction_confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		
		#img_bytes = file.read()
		# save file to disk
		extension = os.path.splitext(file.filename)[1]
		f_name = str(uuid.uuid4()) + extension
		uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f_name)
		file.save(uploaded_image_path)
		
		prediction_name, _ = get_prediction(uploaded_image_path)

		return render_template('index.html', image_upload=f_name, name=prediction_name.lower())

	return render_template('index.html', title="tomato-disease-classification")


if __name__ == '__main__':
	app.run(debug=True)