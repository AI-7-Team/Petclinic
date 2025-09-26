import io
import os
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify # jsonify 임포트 추가
import torch.nn.functional as F
from torchvision import transforms
import base64
import datetime
from AI.model import build_resnet50 # 사용자 정의 모듈
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 초기 설정 및 환경변수 로딩 ---
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE = {DEVICE}")

app = Flask(__name__, static_url_path='/static')

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
client = OpenAI(api_key=api_key)
print("✅ OpenAI API 키 로딩 완료!")

descriptions = {
    'A4': {
        'title': '농포 (Pustule) / 여드름 (Acne)',
        'description': '농포는 고름(농)을 포함하는 작은 융기입니다. 여드름의 한 형태로 나타날 수 있으며, 모낭의 염증이나 감염으로 인해 발생합니다. 반려동물의 경우 박테리아 감염의 신호일 수 있습니다.',
        'image': '/static/images/result_sample_pustule.png'
    },
    'A5': {
        'title': '미란 (Erosion) / 궤양 (Ulcer)',
        'description': '미란은 피부의 가장 바깥층(표피)만 얕게 손상된 상태로, 보통 흉터 없이 치유됩니다. 궤양은 표피를 넘어 진피까지 깊게 손상된 상태로, 치유 후 흉터가 남을 수 있습니다. 지속적인 핥기, 감염, 화상 등 다양한 원인으로 발생할 수 있습니다.',
        'image': '/static/images/result_sample_eu.png'
    },
    'A6': {
        'title': '결절 (Nodule) / 종괴 (Tumor)',
        'description': '결절은 피부 속이나 아래에 생긴 단단한 덩어리(보통 1cm 미만)를 말합니다. 종괴는 결절보다 더 큰 덩어리를 의미하며, 양성일 수도 악성일 수도 있으므로 반드시 수의사의 정밀 진단이 필요합니다. 염증, 감염, 종양 등 심각한 원인의 가능성이 있습니다.',
        'image': '/static/images/result_sample_nt.png'
    },
    'A7': {
        'title': '무증상',
        'description': '탐지된 증상으로서는 발견된 것이 없습니다.',
        'image': '/static/images/result_sample_healthy.png'
    }
}

# --- 2. ResNet 모델 로딩 ---
MODEL_PATH = 'models_train/resnet50_lesion.pth'
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}
model = build_resnet50(num_classes=len(class_to_idx))
model.load_state_dict(checkpoint['model'])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("✅ ResNet 모델 로딩 완료!")

# --- 3. 라우트(Routes) 설정 ---

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
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()

            predicted_class = idx_to_class[predicted_idx]
            confidence = probabilities[0][predicted_idx].item() * 100

            result_description = descriptions.get(predicted_class, {
                'title': '알 수 없는 증상',
                'description': '데이터베이스에 없는 증상입니다.',
                'image': ''
            })
            
            # [수정됨] gpt_response를 직접 생성하지 않고, 템플릿에 기본 정보만 전달
            encoded_img_data = base64.b64encode(file_bytes).decode('utf-8')
            return render_template('result.html',
                                   prediction=predicted_class,
                                   confidence=f"{confidence:.2f}%",
                                   img_data=encoded_img_data,
                                   description_data=result_description
                                   )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("detect.html", error="분석 중 오류가 발생했습니다.")

    return render_template("detect.html")

# [추가됨] JavaScript fetch 요청을 처리하기 위한 GPT 응답 생성 라우트
@app.route('/gpt', methods=['POST'])
def get_gpt_response():
    try:
        data = request.get_json()
        predicted_class = data.get('predicted_class')

        if not predicted_class:
            return jsonify({'error': '예측된 클래스 정보가 없습니다.'}), 400

        result_description = descriptions.get(predicted_class, {})
        predicted_class_title = result_description.get('title')
        gpt_response_message = ""

        if predicted_class == "A7" or predicted_class_title == "무증상":
            gpt_response_message = "정상 소견으로 보입니다. 사진 상으로는 특별한 피부 문제가 발견되지 않았습니다. 만약 반려동물이 다른 이상 행동을 보인다면 수의사와 상담해보시는 것을 권장합니다."
        else:
            disease_name_for_prompt = predicted_class_title.split('(')[0].strip() if predicted_class_title else "알 수 없는 증상"
            system_prompt = "당신은 반려동물 피부질환에 대해 설명하는 AI 전문가입니다. 보호자가 이해하기 쉽도록 친절하고 상세한 설명을 제공해주세요."
            user_prompt = f"""
            반려동물 피부에서 '{disease_name_for_prompt}' 증상이 발견되었습니다.
            아래 항목을 포함하여 보호자를 위해 200자 내외로 설명을 작성해주세요.
            - 마크다운, HTML, 특수문자(*, ~, ` 등)를 사용하지 말고, 순수한 텍스트로만 답변해주세요.
            - 예상 원인
            - 가정에서의 관리 방법
            - 동물병원 방문 권장 여부 및 그 이유
            """
            
            try:
                response = client.chat.completions.create(
                    model="ft:gpt-4.1-2025-04-14:personal:111111:CJtQKVGL",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                gpt_response_message = response.choices[0].message.content.strip()
            except Exception as api_error:
                print(f"Error calling OpenAI API: {api_error}")
                gpt_response_message = "AI 추가 설명을 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요."
        
        return jsonify({'message': gpt_response_message})

    except Exception as e:
        print(f"Error in /gpt route: {e}")
        return jsonify({'error': '서버 처리 중 오류가 발생했습니다.'}), 500


# --- 이하 라우트는 기존과 동일 ---
@app.route('/community')
def community():
    # ...
    return render_template('community.html', posts=[])

# ... (다른 라우트들) ...

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)