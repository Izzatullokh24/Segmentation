import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Load the YOLOv8 model
try:
    model = YOLO('best.pt')  # Ensure the path to your trained YOLOv8 model weights is correct
except FileNotFoundError:
    st.error("Model weights file 'best.pt' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# Predefined colors for each class
COLORS = {
    "high-pneumonia": [255, 0, 0],    # Red
    "low-pneumonia": [0, 255, 0],     # Green
    "no-pneumonia": [0, 0, 255]       # Blue
}

# Define a constant image size
IMAGE_SIZE = (640, 640)  # Width, Height

def segment_image(image):
    results = model(image)
    return results

def display_segmented_image(image, results):
    detected_classes = set()
    if results and results[0].masks is not None and results[0].boxes is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        for mask, box, class_id in zip(masks, boxes, class_ids):
            class_name = model.names[int(class_id)]
            detected_classes.add(class_name)

        # If low-pneumonia or high-pneumonia is detected, ignore no-pneumonia masks
        if 'low-pneumonia' in detected_classes or 'high-pneumonia' in detected_classes:
            detected_classes.discard('no-pneumonia')

        for mask, box, class_id in zip(masks, boxes, class_ids):
            class_name = model.names[int(class_id)]
            if class_name not in detected_classes:
                continue
            mask = mask.astype(np.uint8)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = COLORS.get(class_name, [255, 255, 255])  # Default to white if class not found
            image[mask_resized == 1] = color
            x1, y1, x2, y2 = box[:4].astype(int)
            # Draw a rectangle behind the text for better visibility
            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x1
            text_y = y1 - 10
            cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)
            cv2.putText(image, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Draw the bounding box rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    else:
        st.warning("No objects detected.")

    # Resize the final image to the constant size
    resized_image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    
    return resized_image, detected_classes


st.title('X-ray Segmentation Project')

# Add an author section in the sidebar
st.sidebar.title('About the Author')
st.sidebar.markdown("""
    **Author Name**: Makhammadjonov Izzatullokh 
    **Email**: izzatullokhm@gmail.com  
    <a href="https://github.com/Izzatullokh24" target="_blank"><i class="fab fa-github"></i> GitHub</a>  
    <a href="https://www.linkedin.com/in/izzatullokh-makhammadjonov-242042195/" target="_blank"><i class="fab fa-linkedin"></i> LinkedIn</a>  
    <style>
        .fab {
            font-size: 24px;
            margin-right: 10px;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    Izzatullokh is a machine learning engineer with a passion for computer vision and deep learning.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    try:
        if file_type == 'image':
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = segment_image(image)
            segmented_image, detected_classes = display_segmented_image(image, results)
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

             # Display pneumonia information
            if detected_classes:
                st.subheader('Diagnosis:')
                diagnosis_html = ""
                if 'high-pneumonia' in detected_classes:
                    diagnosis_html += '<p style="color: red; font-size: 20px;">The patient has <strong>high pneumonia</strong>.</p>'
                if 'low-pneumonia' in detected_classes:
                    diagnosis_html += '<p style="color: green; font-size: 20px;">The patient has <strong>low pneumonia</strong>.</p>'
                if 'no-pneumonia' in detected_classes:
                    diagnosis_html += '<p style="color: blue; font-size: 20px;">The patient does <strong>not have pneumonia</strong>.</p>'
                st.markdown(diagnosis_html, unsafe_allow_html=True)
            else:
                st.write('No pneumonia detected.')

        elif file_type == 'video':
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            segmented_frames = segment_video(video_path)
            stframe = st.empty()
            for frame, results in segmented_frames:
                segmented_frame, detected_classes = display_segmented_image(frame, results)
                stframe.image(segmented_frame, channels="BGR")
            os.remove(video_path)

        else:
            st.error("Unsupported file format. Please upload a jpg, jpeg, png, mp4, or avi file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a file to proceed.")
