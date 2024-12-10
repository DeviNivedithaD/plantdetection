import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Load the trained plant disease recognition model
model = tf.keras.models.load_model("trained_model.keras")

# Load the saved known_leaf_features from the .npy file
try:
    known_leaf_features = np.load("known_leaf_features.npy", allow_pickle=True)
except FileNotFoundError:
    st.error("The file 'known_leaf_features.npy' was not found. Ensure it's in your Git repository.")
    st.stop()

# Class names for the model
class_names = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    # Add all the class names corresponding to your trained model
]

# TensorFlow Model Prediction
def model_prediction(test_image):
    """
    Predicts the disease class for the given image using the trained model.
    """
    image = load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = preprocess_input(np.array([input_arr]))  # Preprocess image for prediction
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence  # Return predicted class index and confidence score

# Function to extract features from an image
def extract_features(image_path):
    """
    Extracts deep features from the given image using VGG16.
    """
    image = load_img(image_path, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = preprocess_input(input_arr)  # Preprocess image for feature extraction
    features = feature_extractor.predict(input_arr)
    return features

# Function to validate whether the image resembles a leaf/plant
def validate_leaf_image(image_path):
    """
    Validates if the uploaded image is likely to be a plant/leaf image using feature similarity.
    """
    features = extract_features(image_path)
    similarity_scores = [np.linalg.norm(features - leaf_feature) for leaf_feature in known_leaf_features]
    similarity_threshold = 1.0  # Adjust this threshold based on experimentation
    return min(similarity_scores) <= similarity_threshold

# Dictionary of cures for each disease
disease_cures = {
    'Apple__Apple_scab': "Apply fungicides and improve air circulation.",
    'Apple_Black_rot': "Remove infected leaves and apply fungicides.",
    'Apple_Cedar_apple_rust': "Use resistant varieties and apply fungicides.",
    'Apple__healthy': "No action needed.",
    # Add cures for other diseases...
}

# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("th.jpg", use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Upload an image of a plant, and our system will analyze it to detect diseases.
    ### Get Started
    Click on the Disease Recognition page in the sidebar to upload an image.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of 87K RGB images of healthy and diseased crop leaves, categorized into 38 classes.
    """)
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, width=400, use_container_width=True)
        with open("temp_image.jpg", "wb") as f:
            f.write(test_image.getbuffer())
        if st.button("Predict"):
            st.snow()
            if not validate_leaf_image("temp_image.jpg"):
                st.warning("The uploaded image does not resemble a plant or leaf. Please upload a valid plant image.")
            else:
                st.write("Processing...")
                result_index, confidence = model_prediction("temp_image.jpg")
                confidence_threshold = 0.7  # Adjust based on your model's behavior
                if confidence < confidence_threshold:
                    st.warning("The model is not confident about this prediction.")
                    st.write("Predicted Class: Unknown")
                else:
                    predicted_class = class_names[result_index]
                    st.success(f"Model predicts it's {predicted_class}")
                    cure = disease_cures.get(predicted_class, "No information available for this disease.")
                    st.write(f"Recommended Cure: {cure}")
