import streamlit as st
import os
from fireworks_1 import get_image_description

st.title("Image Upload and Description Test")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "ppm"])

if uploaded_file is not None:
    # Save uploaded file to a temporary path
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Get image description
    description = get_image_description(temp_image_path)
    st.write(f"Image Description: {description}")
