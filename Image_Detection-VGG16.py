import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Open file dialoge to select an image
def select_image():
    root = tk.Tk()
    root.withdraw() 

    file_path = filedialog.askopenfilename()

    return file_path

# Load image to predict top 5 object
img_path = select_image()
if img_path:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Load VGG16
    model = VGG16(weights='imagenet')

    # Predicting
    predictions = model.predict(x)

    # Decode and print the top-5 predicted classes
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")
else:
    print("No image selected.")