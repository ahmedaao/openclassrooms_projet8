# Import packages
import io
import base64
import json
import pickle
import requests
import streamlit as st
from PIL import Image
from src.config import color_map


def app():
    st.title("Image Segmentation App")

    # Summary of the app functionality
    st.write(
        """This basic application is a POC (Proof Of Concept) which take image as input and generate a segmented image
        thanks to a Deep Learning model embedded into a API"""
    )

    st.sidebar.write("## Upload and download :gear:")

    col1, col2 = st.columns(2)
    # Load the image using file_uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image",
                                             type=["png", "jpg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        # Resize the image to input VGG16 (224, 224, 3)
        original_image = original_image.resize((224, 224))
        # Convert the image to bytes
        col1.write("Original Image :camera:")
        col1.image(original_image)

        # Serialize original image to send it to FastAPI
        buffered = io.BytesIO()  # Create a BytesIO object, which is an in-memory file-like object that can be used to store and manipulate binary data.
        original_image.save(buffered, format="PNG")  # Save original image to the buffered BytesIO object in the PNG format
        img_bytes = buffered.getvalue()  # Retrive the binary data stored in the buffered BytesIO object
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Convert binary data to a string of printable ASCII characters

        # Send to the FastAPI
        response = requests.post("http://localhost:8000/segment_image",
                                 json={"image": img_base64},
                                 timeout=120)

        # Deserialize the image returned by FastAPI
        returned_image_base64 = response.json()["image"]
        returned_image_bytes = base64.b64decode(returned_image_base64)
        returned_image = Image.open(io.BytesIO(returned_image_bytes))

        # Display the image returned by FastAPI
        col2.write("Segmented Image :wrench:")
        col2.image(returned_image)

    else:
        st.write("No image uploaded yet.")


if __name__ == "__main__":
    app()
