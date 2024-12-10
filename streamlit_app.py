import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load the trained plant disease recognition model
model = tf.keras.models.load_model("trained_model.keras")

# Class names for the model
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry_(including_sour)__Powdery_mildew',
    'Cherry_(including_sour)__healthy', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)__Common_rust_', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__healthy',
    'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy',
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
    'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot',
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato__healthy'
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
    'Corn_(maize)__Common_rust_': "Use resistant varieties and fungicides.",
    'Corn_(maize)__Northern_Leaf_Blight': "Remove infected debris and apply fungicides.",
    'Corn_(maize)__healthy': "No action needed.",
    'Grape__Black_rot': "Remove infected leaves and apply fungicides.",
    'Grape__Esca_(Black_Measles)': "Prune infected vines and improve drainage.",
    'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides and practice crop rotation.",
    'Grape__healthy': "No action needed.",
    'Orange__Haunglongbing_(Citrus_greening)': "Remove infected trees and control psyllids.",
    'Peach___Bacterial_spot': "Use resistant varieties and apply copper-based fungicides.",
    'Peach__healthy': "No action needed.",
    'Pepper,_ bell__Bacterial_spot': "Remove infected plants and apply copper fungicides.",
    'Pepper,_bell__healthy': "No action needed.",
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
    'Tomato_Late_blight': "Use resistant varieties and apply fungicides.",
    'Tomato__Leaf_Mold': "Improve air circulation and use fungicides.",
    'Tomato__Septoria_leaf_spot': "Remove infected leaves and apply fungicides.",
    'Tomato__Spider_mites Two-spotted_spider_mite': "Use miticides and increase humidity.",
    'Tomato__Target_Spot': "Remove infected leaves and apply fungicides.",
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control aphids.",
    'Tomato__Tomato_mosaic_virus': "Remove infected plants and control aphids.",
    'Tomato__healthy': "No action needed."
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
    Click on the *Disease Recognition* page in the sidebar to upload an image.
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
        # Save the uploaded image temporarily
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(test_image.getbuffer())
        
        # Display the uploaded image
        uploaded_image = Image.open(image_path)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Directly predict the disease
        predicted_index, confidence = model_prediction(image_path)
        disease_name = class_names[predicted_index]
        st.write(f"Predicted Disease: {disease_name} with confidence: {confidence:.2f}")
        st.write("Recommended Cure:", disease_cures.get(disease_name, "No cure available."))
