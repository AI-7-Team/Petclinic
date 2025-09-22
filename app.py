import io
import os
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
import torch.nn.functional as F
from torchvision import transforms
import base64
import datetime
from AI.model import build_resnet50
from openai import OpenAI
from dotenv import load_dotenv
import json

# --- 1. 초기 설정 및 환경변수 로딩 ---
load_dotenv() # .env 파일에서 환경변수 로드
DEVICE = "DEVICE = CUDA" if torch.cuda.is_available() else "DEVICE = CPU"
print(DEVICE)
app = Flask(__name__, static_url_path='/static')

# OpenAI API 키 로딩
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

client = OpenAI(api_key=api_key) # OpenAI 클라이언트 초기화
print("✅ OpenAI API 키 로딩 완료!")

# 예: GPT API로부터 받은 JSON 응답 문자열
#gpt_api_response = '{"summary": "이 내용은 요약입니다.", "advice": "이렇게 하세요."}'
#gpt_response = json.loads(gpt_api_response)  # 문자열 → 딕셔너리# 1. base64 라이브러리 추가


# --- 2. 데이터 및 모델 설정 ---

# 각 증상별 설명 데이터
descriptions = {
    'A4_농포_여드름': {
        'title': '농포 (Pustule) / 여드름 (Acne)',
        'description': '농포는 고름(농)을 포함하는 작은 융기입니다. 여드름의 한 형태로 나타날 수 있으며, 모낭의 염증이나 감염으로 인해 발생합니다. 반려동물의 경우 박테리아 감염의 신호일 수 있습니다.',
        'image': '/static/images/result_sample_pustule.png'
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

# ResNet 모델 로딩
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models_train/resnet50_lesion.pth'

class_to_idx = {
    'A4_농포_여드름': 0,
    'A5_미란_궤양': 1,
    'A6_결절_종괴': 2,
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = build_resnet50(num_classes=len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

            buffered = io.BytesIO(file_bytes)
            encoded_img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

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

            # GPT 호출
            predicted_class_name = result_description["title"]

            if predicted_class_name == "H":  # 건강(H) 클래스가 있다면
                gpt_message = "정상 소견으로 보입니다. 특별한 증상이 없다면 안심하셔도 좋습니다."
            else:
                system_prompt = (
                    "당신은 20년 경력의 유능하고 친절한 수의사입니다. "
                    "보호자가 이해하기 쉽도록 반려견의 증상에 대해 설명하는 역할을 맡았습니다. "
                    "아래 질병에 대해 예상 원인, 가정에서의 관리법, 그리고 동물병원 방문 권장 여부를 "
                    "포함하여 200자 내외로 상세하고 친절하게 설명해주세요."
                )
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"질병: {predicted_class_name}"}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                gpt_response = completion.choices[0].message.content
                print(gpt_response)


            # 결과 반환
            return render_template('result.html',
                                   prediction=predicted_class,
                                   confidence=f"{confidence:.2f}%",
                                   img_data=encoded_img_data,
                                   description_data=result_description,
                                   gpt_response=gpt_response
                                   )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("detect.html", error="분석 중 오류가 발생했습니다.")

    return render_template("detect.html")

# 커뮤니티 라우트
@app.route('/community')
def community():
    """커뮤니티 게시글 목록을 보여주는 페이지입니다."""
    dummy_posts = [
        {'id': 1, 'title': '첫 번째 글입니다', 'author': '관리자', 'create_date': datetime.datetime(2025, 9, 21), 'views': 15},
        {'id': 2, 'title': 'Flask 게시판 만들기', 'author': '김코딩', 'create_date': datetime.datetime(2025, 9, 20), 'views': 42},
    ]
    return render_template('community.html', posts=dummy_posts)

@app.route('/post/<int:post_id>/')
def detail(post_id):
    """개별 게시글의 상세 내용을 보여주는 페이지입니다."""
    return f"게시글 상세 페이지입니다. (ID: {post_id})"

@app.route('/post/create/')
def create():
    """새로운 게시글을 작성하는 페이지를 보여줍니다."""
    return "게시글 작성 페이지입니다."

# 병원 지도 및 404 에러 핸들러
@app.route('/hospital')
def hospital():
    return render_template("hospital.html")

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# --- 4. 앱 실행 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)