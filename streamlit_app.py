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
    input_arr = input_arr / 255.0  # Normalize the image if the model was trained on normalized images
    predictions = model.predict(input_arr)
    
    # Debugging: Print predictions
    print("Raw predictions:", predictions)
    
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence  # Return index of max element and confidence

# Dictionary of cures for each disease
disease_cures = {
    # ... (same as before)
}

# Sidebar
st.sidebar.title("Dashboard ")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    ...
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                ...
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
            confidence_threshold = 0.5  # Lowered threshold for testing
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                          'Blueberry__healthy', 'Cherry_(including_sour)__Powdery_mildew',
                          'Cherry_(including_sour)__healthy', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)__Common_rust_', ' Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__healthy',
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
                input_arr = input_arr / 255.0  # Normalize the image

                # Generate augmented images
                augmented_images = datagen.flow(input_arr, batch_size=1)

                # Display some augmented images
                st.write("Augmented Images:")
                for i in range(5):  # Display 5 augmented images
                    augmented_image = next(augmented_images)[0].astype('float32')
                    st.image(augmented_image, caption=f'Augmented Image {i + 1}', use_column_width=True) 
