import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Load the pre-trained VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Load the model once at the start
model = tf.keras.models.load_model("trained_model.keras")

# Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        # Open the image using PIL
        image = Image.open(test_image)

        # Resize the image to 128x128 and convert to RGB
        image = image.resize((128, 128))
        image = image.convert("RGB")

        # Convert the image to array
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        input_arr = input_arr / 255.0  # Normalize to [0, 1]

        # Debugging: Print shape of input array
        print("Input Array Shape:", input_arr.shape)

        predictions = model.predict(input_arr)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]

        # Debugging: Print predictions and confidence
        print("Predictions:", predictions)  # Log predictions
        print("Predicted Index:", predicted_index)  # Log predicted index
        print("Confidence:", confidence)  # Log confidence

        return predicted_index, confidence  # return index of max element and confidence
    except Exception as e:
        st.error("Error processing image: {}".format(e))
        return None, None

# Function to extract features from an image
def extract_features(image_path):
    image = load_img(image_path, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    features = feature_extractor.predict(input_arr)
    return features

# Function to calculate similarity
def calculate_similarity(features1, features2):
    return np.linalg.norm(features1 - features2)

# Dictionary of cures for each disease
disease_cures = {
    'Apple__Apple_scab': "Apply fungicides and improve air circulation.",
    'Apple_Black_rot': "Remove infected leaves and apply fungicides.",
    'Apple_Cedar_apple_rust': "Use resistant varieties and apply fungicides.",
    'Apple__healthy': "No action needed.",
    'Blueberry__healthy': "No action needed.",
    'Cherry_(including_sour)__Powdery_mildew': "Use sulfur based fungicides.",
    'Cherry_(including_sour)__healthy': "No action needed.",
    'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot': "Apply fungicides and rotate crops.",
    'Corn_(maize)_Common_rust': "Use resistant varieties and fungicides.",
    'Corn_(maize)__Northern_Leaf_Blight': "Remove infected debris and apply fungicides.",
    'Corn_(maize)__healthy': "No action needed.",
    'Grape__Black_rot': "Remove infected leaves and apply fungicides.",
    'Grape_Esca(Black_Measles)': "Prune infected vines and improve drainage.",
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)': "Use fungicides and practice crop rotation.",
    'Grape__healthy': "No action needed.",
    'Orange_Haunglongbing(Citrus_greening)': "Remove infected trees and control psyllids.",
    'Peach___Bacterial_spot': "Use resistant varieties and apply copper-based fungicides.",
    'Peach__healthy': "No action needed.",
    'Pepper,bell_Bacterial_spot': "Remove infected plants and apply copper fungicides.",
    'Pepper,bell_healthy': "No action needed.",
    'Potato__Early_blight': "Apply fungicides and practice crop rotation.",
    'Potato_Late_blight': "Use resistant varieties and apply fungicides.",
    'Potato__healthy': "No action needed.",
    'Raspberry__healthy': "No action needed.",
    'Soybean_healthy': "No action needed.",
    'Squash__Powdery_mildew': "Use sulfur-based fungicides and improve air circulation.",
    'Strawberry__Leaf_scorch': "Use resistant varieties and improve drainage.",
    'Strawberry__healthy': "No action needed.",
    'Tomato__Bacterial_spot': "Remove infected plants and apply copper fungicides.",
    'Tomato__Early_blight': "Apply fungicides and practice crop rotation.",
    'Tomato_Late_b light': "Use resistant varieties and apply fungicides.",
    'Tomato__Leaf_Mold': "Improve air circulation and use fungicides.",
    'Tomato__Septoria_leaf_spot': "Remove infected leaves and apply fungicides.",
    'Tomato__Spider_mites Two-spotted_spider_mite': "Use miticides and increase humidity.",
    'Tomato__Target_Spot': "Remove infected leaves and apply fungicides.",
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control aphids.",
    'Tomato__Tomato_mosaic_virus': "Remove infected plants and control aphids.",
    'Tomato__healthy': "No action needed."
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    ...
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                ...
                """)

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if (test_image is not None):
        if (st.button("Show Image")):
            # Open the image using PIL for display
            image = Image.open(test_image)
            st.image(image, width=400, use_container_width=True)

        # Predict button
        if (st.button("Predict")):
            st.write("Our Prediction")
            result_index, confidence = model_prediction(test_image)
            confidence_threshold = 0.7  # Set a threshold for confidence
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                          'Blueberry__healthy', 'Cherry_(including_sour)__Powdery_mildew',
                          'Cherry_(including_sour)__healthy', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize )__Common_rust_', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__healthy',
                          'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape__healthy', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy',
                          'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
                          'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
                          'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot',
                          'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
                          'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
                          'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                          'Tomato__healthy']
            predicted_class = class_name[result_index]

            if confidence < confidence_threshold:
                st.warning("The model is not confident about this prediction. It may be an unknown disease or not in the database.")
                st.write("Predicted Class: Unknown")
            else:
                st.success("Model is predicting it's a {}".format(predicted_class))
                # Displaying the cure
                cure = disease_cures.get(predicted_class, "NO INFORMATION AVAILABLE FOR THIS DISEASE.")
                st.write("Recommended Cure: {}".format(cure))
