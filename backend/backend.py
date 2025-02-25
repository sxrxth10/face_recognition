from fastapi import FastAPI, File, UploadFile, Response
from recognition import recognize_faces_in_image, recognize_faces_in_video
from io import BytesIO
from PIL import Image

app = FastAPI()


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Read file into memory
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Process the image (pass the PIL image instead of a file path)
    processed_img = recognize_faces_in_image(img)

    # Convert processed image to bytes
    img_bytes = BytesIO()
    processed_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return Response(content=img_bytes.getvalue(), media_type="image/jpeg")


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    processed_video_bytes = recognize_faces_in_video(video_bytes)

    return Response(content=processed_video_bytes, media_type="video/x-msvideo")


