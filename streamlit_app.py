import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    # Resize the image to 128x128
    image = test_image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # return predictions instead of index

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
    'Pepper,_bell__Bacterial_spot': "Remove infected plants and apply copper fungicides.",
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
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"
    st.image(image_path, width=None)  # Use 'width=None' for full width
    
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
    
    # Option to take a photo using the camera
    test_image = st.camera_input("Take a picture of the plant:")
    
    # Option to upload an image from the file system
    if test_image is None:
        test_image = st.file_uploader("Or choose an Image:")
    
    if test_image is not None:
        # Open the image using PIL
        image = Image.open(test_image)
        
        # Compress the image to 128x128
        image = image.resize((128, 128))

        if (st.button("Show Image")):
            st.image(image, width=400)  # Adjusted to use width parameter
        
        # Predict button
        if (st.button("Predict")):
            st.write("Our Prediction")
            predictions = model_prediction(image)
            result_index = np.argmax(predictions)
            confidence = predictions[0][result_index]  # Get confidence of the prediction
            
            # Reading Labels
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
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
                          'Tomato__healthy']
            predicted_class = class_name[result_index]
            
            # Check confidence threshold
            if confidence < 0.5:  # Adjust threshold as needed
                st.warning(" The model is not confident enough in its prediction. Please ensure the image is of a plant leaf.")
            else:
                st.success("Model is predicting it's a {}".format(predicted_class))
                
                # Displaying the cure or no info available
                if predicted_class in disease_cures:
                    cure = disease_cures.get(predicted_class)
                    st.write("Recommended Cure: {}".format(cure))
                else:
                    st.write("No information available for this disease.")
