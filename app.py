import io
import torch
from PIL import Image
from flask import Flask, render_template, request
import torch.nn.functional as F
from torchvision import transforms
import base64  # 1. base64 라이브러리 추가

from AI.model import build_resnet50

app = Flask(__name__, static_url_path='/static')

# 2. 각 증상별 설명 데이터 추가
descriptions = {
    'A4_농포_여드름': {
        'title': '농포 (Pustule) / 여드름 (Acne)',
        'description': '농포는 고름(농)을 포함하는 작은 융기입니다. 여드름의 한 형태로 나타날 수 있으며, 모낭의 염증이나 감염으로 인해 발생합니다. 반려동물의 경우 박테리아 감염의 신호일 수 있습니다.',
        'image': '/static/images/result_sample_pustule.png' # 예시 이미지 경로
    },
    'A5_미란_궤양': {
        'title': '미란 (Erosion) / 궤양 (Ulcer)',
        'description': '미란은 피부의 가장 바깥층(표피)만 얕게 손상된 상태로, 보통 흉터 없이 치유됩니다. 궤양은 표피를 넘어 진피까지 깊게 손상된 상태로, 치유 후 흉터가 남을 수 있습니다. 지속적인 핥기, 감염, 화상 등 다양한 원인으로 발생할 수 있습니다.',
        'image': '/static/images/result_sample_eu.png'
    },
    'A6_결절_종괴': {
        'title': '결절 (Nodule) / 종괴 (Tumor)',
        'description': '결절은 피부 속이나 아래에 생긴 단단한 덩어리(보통 1cm 미만)를 말합니다. 종괴는 결절보다 더 큰 덩어리를 의미하며, 양성일 수도 악성일 수도 있으므로 반드시 수의사의 정밀 진단이 필요합니다. 염증, 감염, 종양 등 심각한 원인의 가능성이 있습니다.',
        'image': '/static/images/result_sample_nt.png'
    }
}

# --- 1. ResNet 모델 로딩 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models_train/resnet50_lesion.pth'

class_to_idx = {
    'A4_농포_여드름': 0,
    'A5_미란_궤양': 1,
    'A6_결절_종괴': 2,
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = build_resnet50(num_classes=len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ ResNet 모델 로딩 완료!")

# --- 2. 라우트(Routes) 설정 ---

@app.route('/')
def index():
    return render_template("about.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("detect.html", error="이미지 파일을 선택해주세요.")
        
        try:
            file_bytes = file.read()
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            
            buffered = io.BytesIO(file_bytes)
            encoded_img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
            
            predicted_class = idx_to_class[predicted_idx]
            confidence = probabilities[0][predicted_idx].item() * 100
            
            # 3. 예측된 클래스에 해당하는 설명 가져오기
            result_description = descriptions.get(predicted_class, {
                'title': '알 수 없는 증상',
                'description': '데이터베이스에 없는 증상입니다.',
                'image': ''
            })

            # 결과 페이지로 모든 정보 전달
            return render_template('result.html', 
                                   prediction=predicted_class,
                                   confidence=f"{confidence:.2f}%",
                                   img_data=encoded_img_data,
                                   description_data=result_description)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("detect.html", error="분석 중 오류가 발생했습니다.")
            
    return render_template("detect.html")

@app.route('/hospital')
def hospital():
    return render_template("hospital.html")

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)