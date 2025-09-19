[<img width="325" height="339" alt="image" src="https://github.com/user-attachments/assets/b42879a3-a27d-41d7-ba5a-34096da9eb81" />](https://github.com/AI-7-Team/petscan/issues/1#issue-3434759771)
# 반려동물 진단 서비스 = PetScan 
AI 기반 반려견 건강 진단 서비스 – PetScan(팻스캔)
PetScan은 반려견의 일상적인 행동, 증상, 변화 등을 AI가 분석하여 건강 이상 신호를 조기에 감지하고 진단을 도와주는 스마트 헬스케어 서비스.
사진 업로드를 통해 반려견, 반려묘의 상태를 훈련된 AI 모델이 수많은 데이터를 기반으로 질병 가능성을 예측.


# 문제 정의 및 서비스 목표

<img src="https://user-images.githubusercontent.com/67316314/190043543-804e79b1-4de9-4edb-90a7-478688912953.jpeg" title="반려견이 동물병원을 찾는 주요 원인- 20위권">

반려견 피부질환의 조기 진단 어려움으로 보호자가 눈으로 보기에 단순 상처/일시적 트러블인지, 혹은 피부병(곰팡이, 세균, 알러지 등)인지 구분하기 어려운점이 문제라고 생각해서 고안해내 서비스입니다.
병원 방문이 늦어질 경우, 질환이 심각해지고 치료 비용이 증가하기 때문에 펫스캔 서비스를 범용적으로 이용하여 조기 발견하게 하는 취지로 구상하였습니다.



1인 가구 확대로 반려동물을 키우는 가구 수가 증가하면서 반려동물 관련 상품뿐만 아니라 반려동물의 질병 관리 및 케어에 관심이 커지고 있습니다. 반려동물 관련 산업이 성장하며 인공지능을 활용한 펫테크 산업 또한 주목받고 있습니다. 
2018년 농촌진흥청은 반려견의 동물병원 내원 중 예방 접종 외에 가장 큰 원인으로 피부염·습진이라고 발표했습니다. [[출처]](https://www.nias.go.kr/front/soboarddown.do?cmCode=M090814150850297&boardSeqNum=3478&fileSeqNum=2323)
2024년 반려동물 보호·복지 실태조사 자료에서는 국내 등록된 반려동물(반려견 + 반려묘)” 누적 수는 349만 마리 수준.이 중 반려견은 약 343만 4,624마리 반려묘(고양이)는 약 5만 6,983마리로 급증하고 있습니다. (농림축산검역본부가 2025년 5월 발표)

**이에 따라 본 해커톤 7팀은 AI 기술을 이용한 반려동물 피부 질환 탐지 기술에 주목하였고 탐지 기술을 이용해 보호자가 반려동물의 피부질환을 조기 발견한다면 적절한 치료를 받을 수 있으리라 생각하였습니다.**

PetScan은 보호자가 반려동물의 피부 질환 사진을 업로드해서 피부 질환 증상에 대한 AI 탐지 결과를 확인할 수 있습니다.      
탐지 결과에 따라 보호자가 전문병원 치료가 필요하다고 판단할 경우 거주 지역 주변의 동물병원을 전국 병원 지도를 통해 확인함으로써 치료로 연계할 수 있는 서비스를 제공합니다.


> 1. 반려동물의 피부 사진을 업로드하여 어떤 증상인지 AI모델을 황용하여 바로 확인할 수 있습니다. 
> 2. 결과를 확인한 유저가 주변 동물병원의 정보를 지도로 확인할 수 있습니다.
  


## 소개 영상

[![YoutubeVideo](https://user-images.githubusercontent.com/67316314/190053730-d32bb591-5abe-4a8e-93ce-89913581f0ef.png)](https://youtu.be/RowdPnEoWNE)

### 웹사이트 링크

- url : [www.petom.site](http://www.petom.site)


## 아키텍처

<img src="https://user-images.githubusercontent.com/67316314/189929257-e99e8e8f-cb90-4505-b659-d9c95c194023.png"/>   


1. 미란, 결절 증상을 탐지하도록 학습시킨 모델을 petom_weights.pt로 저장하여 프로젝트 내에서 로드하여 사용합니다.    
2. EC2 인스턴스의 Flask 서버가 시작 될 때 모델을 로드하고 사용자가 업로드한 이미지를 byte형식으로 변환하고 이미지 파일로 읽습니다.
3. 탐지 결과를 base64 형식으로 encode한 뒤 utf-8로 decode한 최종 결과를 결과 페이지에 전달하여 사용자가 탐지된 증상을 확인하게 됩니다.

- Colab Notebook 환경에서 AIhub에서 제공하는 반려동물 피부 질환 이미지 데이터를 전처리한 후 YOLOv5의 모델 중 하나인 yolov5n모델에 훈련시켰습니다. 
- 웹의 IP 주소를 petom.site 도메인과 연결시키기 위해 Route 53에 호스팅 영역을 생성해 DNS를 설정했습니다.   


---

## 페이지 구성

1. About

<img src="https://user-images.githubusercontent.com/67316314/189915868-48d07786-6033-495e-932e-cf21669435ee.gif" width="30%"/>   

|                        |  | 
| --------------- | -----------  |
| 경로  | [/templates/about.html](https://github.com/SunTera/Petom/blob/main/templates/about.html)|
| URL   | /, /about  |
| 역할            | Petom에서 사용한 알고리즘과 기대효과, 페이지 구성을 확인할 수 있습니다.|


2. 증상탐지∙결과

<img src="https://user-images.githubusercontent.com/98167266/190056565-5f7b3c41-8701-4726-bb1c-8125c4af75af.gif" width="30%"/>

|                        |  | 
| --------------- | -----------  |
| 경로 (증상탐지) | [/templates/detect.html](https://github.com/SunTera/Petom/blob/main/templates/detect.html)|
| 경로 (결과) | [/templates/result.html](https://github.com/SunTera/Petom/blob/main/templates/result.html)|
| URL   | /detect  |
| 역할            | 증상을 탐지할 이미지를 업로드하면 탐지 결과가 결과 페이지로 전달되어 탐지된 증상을 확인할 수 있습니다.|

3. 병원지도   

<img src="https://user-images.githubusercontent.com/67316314/189904236-bf8f6ae7-c709-4c52-8cd0-71fad4d2c10e.gif" width="30%"/>   

|                        |  | 
| --------------- | -----------  |
| 경로  | [/templates/hospital.html](https://github.com/SunTera/Petom/blob/main/templates/hospital.html)|
| URL   | /hospital  |
| 역할            | 전국 동물병원을 시각화한 지도를 통해 병원의 주소와 전화번호를 확인할 수 있습니다.|

---

## 데이터 출처

- 반려동물 피부 질환 이미지

  > AiHub [반려동물 피부 질환](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=561)

- 전국 동물병원 리스트
  > 공공데이터포털 [동물병원 현황 데이터](https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=TOTAL&keyword=%EB%8F%99%EB%AC%BC%EB%B3%91%EC%9B%90&detailKeyword=&publicDataPk=&recmSe=&detailText=&relatedKeyword=&commaNotInData=&commaAndData=&commaOrData=&must_not=&tabId=&dataSetCoreTf=&coreDataNm=&sort=&relRadio=&orgFullName=&orgFilter=&org=&orgSearch=&currentPage=1&perPage=10&brm=&instt=&svcType=&kwrdArray=&extsn=&coreDataNmArray=&pblonsipScopeCode=)

---

## 커밋 메시지 규칙

1. 문장의 끝에 `.` 를 붙이지 않기

2. 이슈 해결 시 커밋 메시지 끝에 이슈 번호 붙이기

   > 예시: issue1 해결 [#1]

3. 형식

   > [타입]: [내용] [이슈 번호]

   - 예시

     > docs: add 메소드 설명 주석 추가 [#1]
     >
     > refactor: 인증 메소드 수정 [#2]

4. 타입 종류

   > \- docs : 문서 작성
   >
   > \- fix : 버그 대처
   >
   > \- refactor : 코드 수정 / 리팩터링
   >
   > \- chore : 간단한 수정
   >
   > \- feature : 새로운 기능
   >
   > \- test : 테스트 추가

