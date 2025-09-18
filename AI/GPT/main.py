from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# OpenAI API 키 설정 (환경 변수 또는 하드코딩)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/predict/explain', methods=['POST'])
def explain_prediction():
    data = request.get_json()

    if not data or 'predicted_class' not in data or 'probabilities' not in data:
        return jsonify({"error": "Invalid input format."}), 400

    predicted_class = data['predicted_class']
    predicted_prob = data['predicted_class_probability']
    probabilities = data['probabilities']

    # 프롬프트 생성
    prompt = (
        f"다음은 이미지 분류 모델의 예측 결과입니다.\n\n"
        f"예측된 클래스: {predicted_class}\n"
        f"확률: {predicted_prob * 100:.2f}%\n"
        f"전체 클래스 확률 분포:\n"
    )
    for cls, prob in probabilities.items():
        prompt += f"- {cls}: {prob * 100:.2f}%\n"

    prompt += "\n가장 높은 확률의 클래스를 중심으로 이 결과를 설명해 주세요."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 친절한 AI 분석가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        explanation = response['choices'][0]['message']['content']
        return jsonify({"message": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
