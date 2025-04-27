from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        print("Эмоция на фото:", box2D.class_name)
        return box2D.class_name, self.draw(image, boxes2D)

detector = EmotionDetector()

stored_images = {
    "vFG56.png": "happy",
    "vFG112.png": "neutral",
    "XMY-074.png": "surprise",
    "vFG137.png": "sad",
    "XMY-136.png": "angry"
}

@app.post("/")
async def detect_emotion(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    user_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if user_image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
    user_emotion, prediction = detector(user_image)

    prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_user.png", prediction_bgr)

    matching_image = None
    for image_path, emotion in stored_images.items():
        if emotion == user_emotion:
            matching_image = image_path
            break

    return {
        "user_emotion": user_emotion,
        "matching_image": matching_image or ""
    }
