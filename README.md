# MI-CLEAR-LLM 체크리스트 자동 생성 도구

본 프로젝트는 "Automated Generation of MI-CLEAR-LLM Checklists for Studies on Large Language Models in Medical Applications Published in Top Journals" 논문의 구현 전체코드입니다.

## 개요

MI-CLEAR-LLM 체크리스트는 의학 연구에서 대규모 언어 모델(LLM)의 사용을 보고하는 표준화된 방법을 제공합니다. 이 도구는 GPT-4o, o1 등의 LLM을 활용하여 연구 논문에서 MI-CLEAR-LLM 체크리스트 항목에 해당하는 정보를 자동으로 추출합니다.

## 주요 기능

- PDF 형식의 의학 논문에서 텍스트 및 PDF로 부터 추출한 페이지 전체 이미지 기반 Reasoning-Findiongs 분석 수행
- 여러 LLM 모델(OpenAI, Azure OpenAI, Google Gemini) 지원
- 다음과 같은 MI-CLEAR-LLM 항목 추출:
  - LLM 정보 (모델명, 버전, 제조사 등)
  - 확률적 특성 (요청 시도 횟수, 온도 설정 등)
  - 프롬프트 보고 (정확한 철자, 기호, 구두점 등)
  - 프롬프트 사용 (채팅 세션 구조, 쿼리 입력 방법 등)
  - 프롬프트 테스트 및 최적화 방법
  - 테스트 데이터셋 독립성
- CSV 형식의 분석 결과 출력

## 설치 방법

### 요구 사항

- Python 3.9 이상
- 필요한 API 키:
  - OpenAI API 키
  - Azure OpenAI API 키 (Azure 사용 시)
  - Google API 키 (Gemini 사용 시)

### 환경 설정

1. 저장소 복제:
```bash
git clone https://github.com/CORE-BMC/mi-clear-llm.git
cd mi-clear-llm
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. `.env` 파일 생성:
```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

## 사용 방법

1. 분석할 저널 논문 PDF 파일을 `input` 폴더에 저장합니다.

2. 설정 파일 조정 (필요시):
   - `config_txt-mode.yaml`: 텍스트 모드 설정
   - `config_img-mode.yaml`: 이미지 모드 설정

3. 실행:

텍스트 기반 분석:
```bash
python mcllm_txt_mode.py
```

이미지 기반 분석:
```bash
python mcllm_img_mode.py
```

4. 결과는 `output` 폴더에 CSV 파일로 저장됩니다.

## 프로젝트 구조

```
mi-clear-llm/
├── mcllm_txt_mode.py      # 텍스트 기반 분석
├── mcllm_img_mode.py      # 이미지 기반 분석
├── config_txt-mode.yaml   # 텍스트 모드 설정
├── config_img-mode.yaml   # 이미지 모드 설정
├── requirements.txt       # 의존성 패키지
├── .env.example           # 환경 변수 예시
├── input/                 # 입력 PDF 파일 위치
└── output/                # 출력 CSV 파일 저장 위치
```

## 참고 문헌

- "KO et al., 2025. Automated Generation of MI-CLEAR-LLM Checklists for Studies on Large Language Models in Medical Applications Published in Top Journals"

## 라이선스

MIT License

## 코드문의

HH (heohwon@gmail.com)
