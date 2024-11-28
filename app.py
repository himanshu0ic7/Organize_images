import streamlit as st
from ultralytics import YOLO
import os
import shutil
import numpy as np

# Load YOLO classification model
model = YOLO('best.pt')

# Function to classify an image and return the class name
def classify_image(image_path):
    results = model.predict(source=image_path)
    class_id = np.array(results[0].probs.data).argmax()
    class_name = results[0].names[class_id]
    return class_name

# Function to classify and organize images
def organize_images(input_dir, output_dir):
    if not os.path.exists(input_dir):
        st.error(f"The directory `{input_dir}` does not exist.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # Ensure it is a file and an image
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Classify the image
            class_name = classify_image(file_path)

            # Create class directory if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Move the image to its class directory
            new_path = os.path.join(class_dir, file_name)
            shutil.move(file_path, new_path)

    st.success("Images classified and organized successfully!")
    st.write(f"Organized images are stored in `{output_dir}`.")
    st.write(f"All original images in `{input_dir}` have been removed.")

# Streamlit application
def main():
    st.title("Image Classification and Organization")
    st.write("Provide the path to a directory of images, and this app will classify and organize them into class-based folders.")

    input_dir = st.text_input("Enter the path to the directory of images:")
    output_dir = st.text_input("Enter the directory to store organized images:", "organized_images")

    if st.button("Classify and Organize"):
        if input_dir.strip() == "":
            st.error("Please enter a valid directory path.")
        else:
            organize_images(input_dir, output_dir)

if __name__ == "__main__":
    main()
