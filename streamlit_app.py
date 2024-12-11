import streamlit as st
import tensorflow as tf
import numpy as np

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

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    ...
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
 This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if (test_image is not None):
        if (st.button("Show Image")):
            st.image(test_image, width=400, use_column_width=True)
        # Predict button
        if (st.button("Predict")):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
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
            st.success("Model is Predicting it's a {}".format(predicted_class))
            # Display recommended cure
            cure = disease_cures.get(predicted_class, "No specific cure available.")
            st.write("### Recommended Cure:")
            st.write(cure)
