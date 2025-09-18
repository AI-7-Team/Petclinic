import os
import json
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from model import build_resnet50  # 모델은 이미 훈련된 build_resnet50을 가져와 사용
import torch.nn.functional as F  # softmax 사용

"""
curl -X POST -F file=@path" http://127.0.0.1:5000/predict
"""
app = Flask(__name__)

# 하이퍼파라미터
MODEL_PATH = './resnet50_lesion.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_idx = {
    'A4_농포_여드름': 0,
    'A5_미란_궤양': 1,
    'A6_결절_종괴': 2,
}

# 모델 로딩
model = build_resnet50(num_classes=len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def home():
    return "Lesion Classification API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # 이미지 처리
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0).to(DEVICE)

        # 모델 예측
        with torch.no_grad():
            output = model(img)  # raw logits
            probabilities = F.softmax(output, dim=1)  # 확률로 변환
            predicted_class = torch.argmax(probabilities, dim=1)  # 가장 높은 확률을 가진 클래스

        # 확률값을 딕셔너리 형태로 저장
        prob_dict = {list(class_to_idx.keys())[i]: probabilities[0][i].item() for i in range(len(class_to_idx))}

        # 가장 높은 확률 클래스와 해당 확률 값
        predicted_class_name = list(class_to_idx.keys())[predicted_class.item()]
        predicted_class_prob = probabilities[0][predicted_class].item()

        return jsonify({
            'predicted_class': predicted_class_name,
            'predicted_class_probability': predicted_class_prob,
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
