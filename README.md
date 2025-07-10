# VLM-architecture

## Vision Language Model 학습 프로젝트

이 저장소는 Vision Language Model(VLM)의 구조와 작동 원리를 학습하기 위한 종합적인 프로젝트입니다.

## 📁 프로젝트 구조

```
VLM-architecture/
├── VLM_Study_Notebook.ipynb    # 메인 학습 노트북
├── requirements.txt            # 필요한 패키지 목록
├── README.md                   # 프로젝트 설명서
└── LICENSE                     # 라이센스 파일
```

## 🚀 시작하기

### 1. 환경 설정

#### Anaconda 환경 생성
```bash
# 새로운 conda 환경 생성
conda create -n vlm-study python=3.10 -y

# 환경 활성화
conda activate vlm-study

# Jupyter 설치
conda install jupyter notebook ipykernel -y
```

#### 필수 패키지 설치
```bash
# 기본 패키지 설치
pip install torch torchvision transformers datasets pillow requests accelerate

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 2. 노트북 실행
```bash
jupyter notebook VLM_Study_Notebook.ipynb
```

## 📚 학습 내용

### 주요 토픽
1. **VLM 기본 개념** - 멀티모달 AI의 이해
2. **CLIP 모델** - 대조 학습 기반 이미지-텍스트 이해
3. **BLIP 모델** - 이미지 캡셔닝 및 VQA
4. **멀티모달 어텐션** - 크로스 모달 상호작용 메커니즘
5. **실제 응용 사례** - 산업 적용 예시

### 실습 내용
- ✅ CLIP을 사용한 이미지-텍스트 유사도 계산
- ✅ BLIP으로 자동 이미지 캡션 생성
- ✅ 멀티모달 어텐션 메커니즘 구현
- ✅ VLM 모델 성능 비교 분석
- ✅ 어텐션 가중치 시각화

## 🎯 학습 목표

이 프로젝트를 통해 다음을 학습할 수 있습니다:

- VLM의 핵심 아키텍처 이해
- 주요 VLM 모델들(CLIP, BLIP, LLaVA)의 특징 파악
- 멀티모달 데이터 처리 방법 습득
- 실제 VLM 모델 사용법 익히기
- 어텐션 메커니즘의 시각화 및 해석

## 🔧 요구사항

### 시스템 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장, CPU로도 실행 가능)
- 최소 8GB RAM

### 소프트웨어 의존성
- PyTorch 2.0+
- Transformers 4.30+
- Jupyter Notebook
- 기타 패키지는 `requirements.txt` 참조

## 📖 추가 학습 자료

### 논문
- [CLIP: Learning Transferable Visual Representations from Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)

### 유용한 링크
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [BLIP Official Repo](https://github.com/salesforce/BLIP)

## 🤝 기여하기

프로젝트 개선을 위한 기여를 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**Happy Learning! 🎉**