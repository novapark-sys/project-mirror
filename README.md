# 💎 Project Mirror: Synthetic Panel Engine

**Project Mirror**는 과거 설문 데이터를 기반으로 실존 인물을 모사하는 **'합성 패널(Synthetic Panel)'** 시뮬레이션 엔진입니다.

스탠퍼드 대학교의 [*Generative Agents (2023)*](https://arxiv.org/abs/2304.03442) 논문에서 제안한 기억 인출 알고리즘(Retrieval Function)과 성찰(Reflection) 메커니즘을 비즈니스 환경에 맞춰 최적화하였습니다.

---

## 🌟 Key Features (핵심 기능)

### 1. Dual-Layer Brain Architecture 
* **Layer 1. Core Persona (장기 기억/성찰):** 전체 데이터를 관조(Reflection)하여 추출한 불변의 가치관과 성격을 요약합니다.
  * '성찰'에서는 최대한 모든 데이터를 훑는 것이 개인의 특성을 파악하기에 효과적일거라 판단. 하지만 비용, 읽는 속도의 이슈가 있어, 과거부터 현재까지를 특정 chunk_size로 띄엄띄엄 scan하도록 함.
* **Layer 2. Retrieval Memory (단기 기억/증거):** 질문과 관련된 구체적인 과거 행동을 '기억 Scoring System'을 기반으로 찾아냅니다.
  * 질문과 관련된 기억을 모든 데이터를 키워드 검색으로 일부 추출하고, 선별된 후보군에 대해서 최신성 / 중요도 / 적합성 을 기준으로 score를 매기고, 높은 Score를 가진 top 10개의 메모리를 가져온다.

### 2. 기억 Scoring System 
논문의 수식을 그대로 구현하여 가장 적합한 기억을 선별합니다.
* **Recency ($R$):** `created_at` (응답 생성일) 기준 지수 감쇠(Exponential Decay) 적용. ($0.998^{days}$)
* **Importance ($I$):** LLM이 판단한 기억의 중요도 (1~10점).
* **Relevance ($R$):** 질문과 기억 간의 임베딩 벡터 코사인 유사도.

### 3. Explainable AI (Knowledge Graph)
* AI가 왜 그런 답변을 했는지, 어떤 기억을 참조했는지 지식 그래프(Knowledge Graph) 형태로 시각화하여 보여줍니다.

---

## 🛠️ Installation (설치 방법)

이 프로젝트는 **Python 3.9+** 환경에서 동작합니다.

### 1. 레포지토리 클론 및 이동
```bash
git clone [https://github.com/novapark-sys/project-mirror.git](https://github.com/project-mirror.git)
cd project-mirror
