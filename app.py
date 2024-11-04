import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, render_template, request
from PIL import Image
import io
import base64
import pickle
import os

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def detect_deepfake(frame):
    boxes, probs = mtcnn.detect(frame)
    
    if boxes is None:
        return None
    
    x1, y1, x2, y2 = boxes[0].astype(int)
    face = frame[y1:y2, x1:x2]
    
    face = cv2.resize(face, (160, 160))    
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # Normalizes to range [-1, 1]
    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        output = torch.sigmoid(model(face_tensor)).item()
    print(f"Deepfake Score: {output:.2f}")    
    return output

app = Flask(__name__, static_url_path='/static')  

@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' in request.files:
        imagefile = request.files['imagefile']
        image_path = os.path.join("./images", imagefile.filename)
        imagefile.save(image_path)
        image = Image.open(imagefile.stream)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
        face = mtcnn(image)
        if face is None:
            raise Exception('No face detected')

        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        face = face.to(DEVICE).float() / 255.0

        with torch.no_grad():
            output = torch.sigmoid(model(face)).item()
            if output < 0.5:
                prediction = "real"
                confidence = round((1 - output) * 100, 2) 
            else:
                prediction = "fake"
                confidence = round(output * 100, 2)
        return render_template("index.html", prediction=prediction, confidence=confidence, image=image_base64)

    elif 'videofile' in request.files:
        videofile = request.files['videofile']
        video_path = os.path.join("./videos", videofile.filename)
        videofile.save(video_path)

        cap = cv2.VideoCapture(video_path)
        total_score = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            deepfake_score = detect_deepfake(frame)
            if deepfake_score is not None:
                total_score += deepfake_score
                frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            average_score = total_score / frame_count
            if average_score < 0.5:
                prediction = "real"
                confidence = round((1 - average_score) * 100, 2)  
            else:
                prediction = "fake"
                confidence = round(average_score * 100, 2)
            
        else:
            prediction = "real"
            confidence = 100  

        return render_template("index.html", prediction_video=prediction, confidence_video=confidence)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
