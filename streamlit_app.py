import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Load the trained model for prediction
model = tf.keras.models.load_model("trained_model.keras")

# TensorFlow Model Prediction
def model_prediction(test_image):
    # Resize the image to 128x128
    image = load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence  # Return index of max element and confidence

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
st.sidebar.title("Dashboard ")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == Home 
st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
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
elif app_mode == "About":
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
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=400, use_container_width=True)
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index, confidence = model_prediction(test_image)
            confidence_threshold = 0.7  # Set a threshold for confidence
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

            if confidence < confidence_threshold:
                st.warning("The model is not confident about this prediction. It may be an unknown disease or not in the database.")
                st.write("Predicted Class: Unknown")
            else:
                st.success("Model is predicting it's a {}".format(predicted_class))
                # Displaying the cure
                cure = disease_cures.get(predicted_class, "NO INFORMATION AVAILABLE FOR THIS DISEASE.")
                st.write("Recommended Cure: {}".format(cure))

                # Augment the image using ImageDataGenerator
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )

                # Prepare the image for augmentation
                input_arr = img_to_array(load_img(test_image, target_size=(128, 128)))
                input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

                # Generate augmented images
                augmented_images = datagen.flow(input_arr, batch_size=1)

                # Display some augmented images
                st.write("Augmented Images:")
                for i in range(5):  # Display 5 augmented images
                    augmented_image = next(augmented_images)[0].astype('float32')
                    st.image(augmented_image / 255.0, caption=f'Augmented Image {i + 1}', use_column_width=True)
