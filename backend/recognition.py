import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import joblib
import cv2
import tempfile


# Initialize MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=True, post_process=False, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
face_recognition_model = load_model("models/face_recognition_model_updated.h5")

# Load the saved LabelEncoder
label_encoder = joblib.load("models/player_labels.pkl")

def recognize_faces_in_image(img):
    draw = ImageDraw.Draw(img)

    # Detect faces
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Extract face embedding
            face = img.crop((x1, y1, x2, y2))
            face = face.resize((160, 160))
            face_tensor = np.array(face).transpose((2, 0, 1)) / 255.0
            face_tensor = torch.tensor(face_tensor, dtype=torch.float32).unsqueeze(0)

            # Get embeddings and predictions
            face_embedding = model(face_tensor).detach().cpu().numpy()
            prediction = face_recognition_model.predict(face_embedding)
            player_id = np.argmax(prediction)
            predicted_player = label_encoder.inverse_transform([player_id])[0]
            confidence = np.max(prediction)

             # Load a font (adjust the size as needed)
            font_path = "arial.ttf"  # Make sure the font file exists
            font_size = 25  # Increase this for larger text
            font = ImageFont.truetype(font_path, font_size)

            # Draw bounding box and name
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            if confidence > 0.7:
                draw.text((x1, y1 - 30), f"{predicted_player} ({confidence:.2f})", fill="green",font=font)

    return img  # Return the processed image directly





def recognize_faces_in_video(video_bytes):
    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video Writer setup
    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    output_video_path = output_video.name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        # Convert frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        # Detect faces
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Extract face embedding
                face = img.crop((x1, y1, x2, y2))
                face = face.resize((160, 160))
                face_tensor = torch.tensor(np.array(face).transpose((2, 0, 1)) / 255.0, dtype=torch.float32).unsqueeze(0).to(model.device)

                # Get face embedding
                with torch.no_grad():
                    face_embedding = model(face_tensor).cpu().numpy()

                # Predict player using trained model
                prediction = face_recognition_model.predict(face_embedding)
                player_id = np.argmax(prediction)
                predicted_player = label_encoder.inverse_transform([player_id])[0] 
                confidence = np.max(prediction) 

                # font_path = "arial.ttf"  # Make sure the font file exists
                # font_size = 25  # Increase this for larger text
                # font = ImageFont.truetype(font_path, font_size)
                font= ImageFont.load_default(25)


                # Draw bounding box and name
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                if confidence > 0.7:
                    draw.text((x1, y1 - 30), f"{predicted_player} ({confidence:.2f})", fill="green",font=font)


        # Convert PIL image back to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()

    # Read the processed video file as bytes
    with open(output_video_path, "rb") as f:
        processed_video_bytes = f.read()

    return processed_video_bytes  # Return video as bytes