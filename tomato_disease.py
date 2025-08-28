import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models  import load_model
import numpy as np
import streamlit as st
from PIL import Image

model = load_model("tomato_model.keras")

class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']



st.title("Tomato Disease Predictiion")

uploaded_file = st.file_uploader("Upload a tomato's leaf image : ", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None : 
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption = 'Uploaded Image', use_column_width = True)


    img = image.resize((224,224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis =0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"**Prediction:** {predicted_class}")




