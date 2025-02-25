import streamlit as st
import requests
import tempfile
from PIL import Image
from io import BytesIO

BACKEND_URL = "http://35.154.237.199:8000"

st.title("Face Recognition System")

option = st.radio("Choose input type:", ("Image", "Video"))

# Initialize session state variables
if "processing" not in st.session_state:
    st.session_state.processing = False  # Processing flag
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
    st.session_state.file_type = None

# Function to stop backend processing
def stop_processing():
    requests.post(f"{BACKEND_URL}/stop-processing/")  # Send stop signal to backend
    st.session_state.processing = False  # Update frontend state

if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image and not st.session_state.processing and st.session_state.processed_file is None:
        st.session_state.processing = True  # Mark as processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_image.read())
            temp_path = temp.name

        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BACKEND_URL}/process-image/", files=files)

        if response.status_code == 200:
            st.session_state.processed_file = response.content
            st.session_state.file_type = "image"
            st.session_state.processing = False  # Mark as completed

    # Display processed image
    if st.session_state.processed_file:
        image_bytes = BytesIO(st.session_state.processed_file) #open
        processed_image = Image.open(image_bytes)
        st.image(processed_image, caption="Processed Image", use_container_width=True)

        st.download_button(
            "Download Processed Image", 
            st.session_state.processed_file, 
            "processed_image.jpg", 
            "image/jpeg"
        )

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_video and not st.session_state.processing and st.session_state.processed_file is None:
        st.session_state.processing = True  # Mark as processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(uploaded_video.read())
            temp_path = temp.name

        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BACKEND_URL}/process-video/", files=files)

        if response.status_code == 200:
            st.session_state.processed_file = response.content
            st.session_state.file_type = "video"
            st.session_state.processing = False  # Mark as completed

    # Display processed video
    if st.session_state.processed_file:
        st.video(st.session_state.processed_file)

        st.download_button(
            "Download Processed Video", 
            st.session_state.processed_file, 
            "processed_video.avi", 
            "video/x-msvideo"
        )

# Stop Processing Button
if st.session_state.processing:
    if st.button("Stop Processing"):
        stop_processing()
