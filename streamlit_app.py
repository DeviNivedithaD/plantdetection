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
    'Apple___Apple_scab': "Apply fungicides and improve air circulation.",
    'Apple___Black_rot': "Remove infected leaves and apply fungicides.",
    'Apple___Cedar_apple_rust': "Use resistant varieties and apply fungicides.",
    'Apple___healthy': "No action needed.",
    'Blueberry___healthy': "No action needed.",
    'Cherry_(including_sour)___Powdery_mildew': "Use sulfur based fungicides.",
    'Cherry_(including_sour)___healthy': "No action needed.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Apply fungicides and rotate crops.",
    'Corn_(maize)___Common_rust_': "Use resistant varieties and fungicides.",
    'Corn_(maize)___Northern_Leaf_Blight': "Remove infected debris and apply fungicides.",
    'Corn_(maize)___healthy': "No action needed.",
    'Grape___Black_rot': "Remove infected leaves and apply fungicides.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines and improve drainage.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides and practice crop rotation.",
    'Grape___healthy': "No action needed.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees and control psyllids.",
    'Peach___Bacterial_spot': "Use resistant varieties and apply copper-based fungicides.",
    'Peach___healthy': "No action needed.",
    'Pepper,_bell___Bacterial_spot': "Remove infected plants and apply copper fungicides.",
    'Pepper,_bell___healthy': "No action needed.",
    'Potato___Early_blight': "Apply fungicides and practice crop rotation.",
    'Potato___Late_blight': "Use resistant varieties and apply fungicides.",
    'Potato___healthy': "No action needed.",
    'Raspberry___healthy': "No action needed.",
    'Soybean___healthy': "No action needed.",
    'Squash___Powdery_mildew': "Use sulfur-based fungicides and improve air circulation.",
    'Strawberry___Leaf_scorch': "Use resistant varieties and improve drainage.",
    'Strawberry___healthy': "No action needed.",
    'Tomato___Bacterial_spot': "Remove infected plants and apply copper fungicides.",
    'Tomato___Early_blight': "Apply fungicides and practice crop rotation.",
    'Tomato___Late_blight': "Use resistant varieties and apply fungicides.",
    'Tomato___Leaf_Mold': "Improve air circulation and use fungicides.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves and apply fungicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides and increase humidity.",
    'Tomato___Target_Spot': "Remove infected leaves and apply fungicides.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control aphids.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants and control aphids.",
    'Tomato___healthy': "No action needed."
}
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"
    st.image(image_path, use_container_width=True)  # Use 'width=None' for full width
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the **Disease Recognition ** page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *User -Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
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
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
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
