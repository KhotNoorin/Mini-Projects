import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf

# Constants
IMAGE_DIR = r"C:\Users\NOORIN\Documents\HAM10000_images_part_1"  # Path to image folder
METADATA_PATH = r'C:\Users\NOORIN\Documents\HAM10000_metadata.csv'  # Path to metadata CSV
MODEL_FILE = 'dermatology_disease_detection_model.h5'  # Path for saving/loading model

# Load and preprocess images
@st.cache_data
def load_and_resize_images(metadata, image_dir, target_size=(64, 64)):
    images = []  # List to store image data
    labels = []  # List to store corresponding labels
    
    if not os.path.exists(image_dir):
        st.error("Error: The specified image directory does not exist.")
        return None, None

    for index, row in metadata.iterrows():
        image_path = os.path.join(image_dir, row['image_id'] + '.jpg')
        if not os.path.exists(image_path):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        resized_image = cv2.resize(image, target_size)
        images.append(resized_image)
        labels.append(row['dx'])  # 'dx' is the diagnosis label
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Define and compile CNN model
def create_cnn_model(input_shape=(64, 64, 3), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model():
    metadata = pd.read_csv(METADATA_PATH)
    images, labels = load_and_resize_images(metadata, IMAGE_DIR)
    
    if images is None or labels is None:
        st.error("Error: Could not load images.")
        return None
    
    # Normalize images
    images = images / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    # Data Augmentation
    data_augmentation = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
    data_augmentation.fit(X_train)

    # Create and train the model
    model = create_cnn_model()
    history = model.fit(data_augmentation.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test), steps_per_epoch=X_train.shape[0] // 32)

    # Save the model
    model.save(MODEL_FILE)
    
    return model, history, label_encoder

# Function to predict new image
def predict_new_image(model, label_encoder, image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_label = label_encoder.inverse_transform([predicted_class])[0]
    return class_label

# Streamlit App
st.title("Dermatological Disease Detection")
st.write("Say No To Skin Diseases!")
st.write("Check your skin and get instant results within 1 minute.")
st.write("Upload an image of a skin lesion and predict its type.")

# File uploader for image
uploaded_image = st.file_uploader("Upload Image (JPG format)", type=["jpg", "jpeg"])

# Check if model exists and load it
if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    metadata = pd.read_csv(METADATA_PATH)
    label_encoder = LabelEncoder()
    label_encoder.fit(metadata['dx'])
else:
    st.warning("Model not found. Training a new model...")
    model, history, label_encoder = train_model()

# Display the training history
if 'history' in locals():
    st.subheader("Training History")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    
    st.pyplot(fig)

# Predict on uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    image_path = os.path.join("temp_image.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    predicted_class = predict_new_image(model, label_encoder, image_path)
    st.write(f"Predicted Class: **{predicted_class}**")

# Provide option to retrain the model
if st.button("Retrain Model"):
    model, history, label_encoder = train_model()
    st.success("Model retrained successfully!")
