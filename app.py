import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import tempfile
import asyncio
from datetime import datetime
import math
import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

# LangChain & OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List

# --- 1. 설정 및 UI 초기화 ---
st.set_page_config(page_title="Project Mirror", layout="wide")

st.title("💎 Project Mirror: 개인단위 합성패널")
st.markdown("""
    Project Mirror는 과거 설문 데이터를 기반으로 실존 인물을 완벽하게 모사하는 '초개인화 합성 패널(Synthetic Panel)' 시뮬레이션 엔진입니다.
    스탠퍼드 대학교의 Generative Agents (2023) 논문에서 제안한 기억 인출 알고리즘(Retrieval Function)과 성찰(Reflection) 메커니즘을 비즈니스 환경에 맞춰 최적화한 Hybrid RAG 모델입니다.

""")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.divider()
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader("설문 데이터 (CSV)", type=["csv"])
    
    st.divider()
    st.subheader("🎛️ 인출 가중치 (Retrieval Weights)")
    alpha = st.slider("Recency (최신성)", 0.0, 2.0, 1.0, help="최근 기억을 얼마나 중요하게 볼 것인가")
    beta = st.slider("Importance (중요도)", 0.0, 2.0, 1.0, help="인생의 중요한 사건을 얼마나 가중할 것인가")
    gamma = st.slider("Relevance (관련성)", 0.0, 5.0, 3.0, help="질문과 직접 관련된 내용을 얼마나 찾을 것인가")

    if not openai_api_key:
        st.warning("⚠️ OpenAI API 키를 입력해주세요.")
        st.stop()

# --- 2. 핵심 로직 함수 ---

@st.cache_resource
def get_models(api_key):
    # mini 모델 사용
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    return llm, embeddings

def generate_persona_summary(llm, df):
    """
    [Ideal Implementation] 
    전체 데이터를 청크(Chunk)로 나누어 '중간 성찰(Insight)'을 얻고,
    이를 최종적으로 통합하여 '핵심 페르소나'를 추출하는 Map-Reduce 방식
    """
    
    # 1. 데이터를 시간순으로 정렬 (과거 -> 현재 흐름 파악 중요)
    df_sorted = df.sort_values('created_at', ascending=True)
    
    # 2. [Map 단계] 데이터를 50개씩 잘라서 '중간 요약(Mini-Reflection)' 생성
    # (5000개 데이터라면 약 100번의 LLM 호출이 발생하지만, 가장 정확함)
    # *비용/속도 조절을 위해 여기서는 100개 단위로 stride하거나, 중요 데이터만 필터링 할 수도 있습니다.
    # *PoC용으로는 전체를 다 돌리기보다, '1년 단위' 또는 '50개 단위'로 끊는 것을 추천합니다.
    
    chunk_size = 50
    chunks = [df_sorted[i:i + chunk_size] for i in range(0, len(df_sorted), chunk_size * 3)]
    
    # 진행 상황 표시용 (Streamlit 전용)
    progress_text = "전체 생애 데이터를 정독하며 성찰 중입니다... (Map-Reduce)"
    my_bar = st.progress(0, text=progress_text)
    
    intermediate_insights = []
    
    for i, chunk in enumerate(chunks):
        text_blob = ""
        for _, row in chunk.iterrows():
            text_blob += f"- [{row['created_at'].strftime('%Y-%m')}] Q:{row['question']} A:{row['answer']}\n"
            
        # 중간 성찰 프롬프트
        map_prompt = f"""
        Analyze these survey records (a part of a person's life).
        Extract 3-5 key keywords or short sentences regarding their **personality, values, and changes**.
        
        [Records]
        {text_blob}
        """
        # 빠르게 처리하기 위해 비동기로 돌리거나, 여기서 invoke
        insight = llm.invoke(map_prompt).content
        intermediate_insights.append(insight)
        
        # 진행률 업데이트
        my_bar.progress((i + 1) / len(chunks), text=f"{i+1}/{len(chunks)} 구간 분석 중...")
    
    my_bar.empty() # 바 제거

    # 3. [Reduce 단계] 중간 요약들을 모아서 '최종 페르소나' 생성
    # 중간 요약이 너무 길면 이것도 다시 쪼개야 하지만, 보통은 Context Window 내에 들어옵니다.
    all_insights = "\n".join(intermediate_insights)
    
    final_prompt = f"""
    Below are the chronological insights extracted from a person's entire survey history (from past to present).
    Synthesize these into a **comprehensive 'Core Persona'**.
    
    [Chronological Insights]
    {all_insights}
    
    [Instruction]
    1. Identify consistent traits (Immutable Core).
    2. Note any changes in values or lifestyle over time (Evolution).
    3. Summarize into a structured profile in **Korean**:
       - **기본 성향:** (Personality & Tone)
       - **핵심 가치관:** (Values)
       - **소비/라이프스타일:** (Spending & Life)
       - **주요 변화:** (Life trajectory)
    """
    
    return llm.invoke(final_prompt).content

def search_candidates_from_full_data(df, query, limit=30):
    """[Broad Search] 전체 데이터 대상 키워드 1차 필터링"""
    keywords = query.split()
    keywords = [w for w in keywords if len(w) >= 2] # 2글자 이상만
    
    candidates = pd.DataFrame()
    if keywords:
        # 질문이나 답변에 키워드가 포함된 경우
        mask = df['question'].astype(str).str.contains('|'.join(keywords), case=False) | \
               df['answer'].astype(str).str.contains('|'.join(keywords), case=False)
        candidates = df[mask]
    
    # 후보가 부족하면 최신 데이터로 채움 (Context 유지)
    if len(candidates) < limit:
        needed = limit - len(candidates)
        recent_df = df.head(needed)
        candidates = pd.concat([candidates, recent_df]).drop_duplicates()
        
    return candidates.head(limit)

async def evaluate_importance(llm, text):
    """[Importance] 기억의 중요도 평가 (1~10)"""
    try:
        res = await llm.ainvoke(f"Rate importance (1-10) of this memory: '{text}'. Return number only.")
        val = int(''.join(filter(str.isdigit, res.content)))
        return min(max(val, 1), 10)
    except: return 5

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def calculate_scores_and_rank(candidates_df, query, llm, embeddings_model, weights):
    """[Deep Scoring] 논문 수식 적용 ($Score = R + I + R$)"""
    texts = [f"Q: {row['question']} / A: {row['answer']}" for _, row in candidates_df.iterrows()]
    
    # 1. 비동기 병렬 처리: 중요도 평가 & 임베딩
    importance_tasks = [evaluate_importance(llm, t) for t in texts]
    
    importances, query_vector, doc_vectors = await asyncio.gather(
        asyncio.gather(*importance_tasks),
        embeddings_model.aembed_query(query),
        embeddings_model.aembed_documents(texts)
    )
    
    scored_memories = []
    current_time = datetime.now()
    alpha, beta, gamma = weights
    
    for i, (_, row) in enumerate(candidates_df.iterrows()):
        # A. Recency (Exponential Decay)
        # 실제 데이터가 과거 데이터이므로, 현재 시점과의 차이(일 단위) 계산
        days_diff = abs((current_time - row['created_at']).days)
        # 1년(365일) 지나면 점수가 약 0.5배가 되도록 감쇠 (0.998)
        recency = math.pow(0.998, days_diff)
        
        # B. Importance (Normalized 0~1)
        importance = importances[i] / 10.0
        
        # C. Relevance (Cosine Similarity)
        relevance = cosine_similarity(query_vector, doc_vectors[i])
        
        # D. Final Score
        score = (alpha * recency) + (beta * importance) + (gamma * relevance)
        
        scored_memories.append({
            "text": texts[i], 
            "score": score,
            "details": {
                "Recency(최신)": round(recency, 2), 
                "Importance(중요)": round(importance, 2), 
                "Relevance(관련)": round(relevance, 2)
            },
            "created_at": row['created_at']
        })
    
    # 점수순 정렬 후 Top 5 반환
    scored_memories.sort(key=lambda x: x['score'], reverse=True)
    return scored_memories[:10]

def visualize_brain_map(top_memories, persona_summary):
    """[Visualization] 사고 과정 시각화 (Brain Map)"""
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    
    # 중심 노드 (User) - 페르소나 요약 포함
    user_label = "User (Persona)"
    net.add_node(user_label, label=user_label, title=persona_summary, color="#1f77b4", size=30)
    
    # 기억 노드 연결
    for i, mem in enumerate(top_memories):
        mem_id = f"Memory_{i+1}"
        # 점수에 따라 노드 크기 조절
        size = 15 + (mem['score'] * 5)
        
        # 툴팁에 상세 정보 표시
        tooltip = f"Total Score: {mem['score']:.2f}\n\n[Details]\n{mem['details']}\n\n[Content]\n{mem['text']}"
        
        # 색상: 점수가 높을수록 진한 색 (Orange -> Red)
        color = "#ff7f0e" if mem['score'] < 3.0 else "#d62728"
        
        net.add_node(mem_id, label=f"Memory {i+1}", title=tooltip, color=color, size=size)
        
        # 엣지: 두께를 점수에 비례하게
        net.add_edge(user_label, mem_id, width=mem['score'], color="#cccccc")

    # 물리 엔진 설정
    net.force_atlas_2based()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            net.save_graph(f.name)
            return f.name
    except: return None

def generate_final_answer(llm, query, top_memories, persona_summary):
    """[Generation] CoT 기반 답변 생성"""
    memory_text = "\n".join([f"- {m['text']}" for m in top_memories])
    
    prompt = f"""
    You are a simulation of a real person based on survey data.
    
    [Core Persona]
    {persona_summary}
    
    [Retrieved Memories (Top Scored)]
    {memory_text}
    
    [Question]
    {query}
    
    [Instruction]
    1. **Think Step-by-Step:** Analyze how your 'Core Persona' and specific 'Memories' relate to the question.
    2. **Resolve Conflicts:** If recent memories conflict with old persona, follow the recent behavior (refer to dates in memory).
    3. **Tone & Style:** Mimic the tone found in the memories (short/long, formal/casual).
    4. **Answer:** Respond in the first person. Explain your reasoning clearly.
    """
    return llm.invoke(prompt).content

# --- 3. 메인 실행 ---

# 세션 초기화
if 'persona_summary' not in st.session_state:
    st.session_state['persona_summary'] = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp949')
        
    # [필수] created_at을 datetime 객체로 변환 (Recency 계산용)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at', ascending=False)
    else:
        st.error("오류: CSV 파일에 'created_at' 컬럼이 반드시 필요합니다.")
        st.stop()

    # 1. 페르소나 생성 (최초 1회)
    if st.session_state['persona_summary'] is None:
        with st.spinner("🕵️ 전체 데이터를 분석하여 '핵심 페르소나'를 추출 중..."):
            llm_base, _ = get_models(openai_api_key)
            st.session_state['persona_summary'] = generate_persona_summary(llm_base, df)
            st.success("페르소나 분석 완료!")
            
    with st.sidebar.expander("👤 분석된 페르소나", expanded=True):
        st.info(st.session_state['persona_summary'])

    # 2. 인터뷰
    st.subheader("💬 AI 패널 심층 인터뷰")
    
    col_q, col_btn = st.columns([4, 1])
    with col_q:
        query = st.text_input("질문 입력", label_visibility="collapsed", placeholder="예: 이 패널은 새로운 구독 서비스를 신청할까요?")
    with col_btn:
        run_btn = st.button("예측 실행", use_container_width=True)
    
    if query and run_btn:
        llm, embed_model = get_models(openai_api_key)
        
        # A. 검색 & 채점
        with st.spinner("1. 기억 인출 및 정밀 채점 중 ($R+I+R$)..."):
            # 1단계: 전체 데이터 대상 키워드 검색
            candidates = search_candidates_from_full_data(df, query, limit=30)
            
            # 2단계: 논문 공식 적용 (비동기)
            top_memories = asyncio.run(
                calculate_scores_and_rank(candidates, query, llm, embed_model, (alpha, beta, gamma))
            )
            
        # B. 답변 & 시각화
        with st.spinner("2. 답변 생성 및 뇌 구조 시각화 중..."):
            # 3단계: 답변 생성
            answer = generate_final_answer(llm, query, top_memories, st.session_state['persona_summary'])
            # 4단계: 그래프 그리기
            graph_html = visualize_brain_map(top_memories, st.session_state['persona_summary'])
            
        # C. 결과 화면
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🤖 AI 답변")
            st.write(answer)
            st.markdown("---")
            st.caption("💡 우측 그래프는 AI가 답변을 위해 활성화한 '생각의 지도'입니다.")

        with col2:
            st.markdown("### 🧠 사고의 지도 (Brain Map)")
            if graph_html:
                with open(graph_html, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=400)
            
            with st.expander("🔍 기억별 상세 점수표 (Evidence)"):
                for m in top_memories:
                    score_info = f"Score: {m['score']:.2f} (R:{m['details']['Recency(최신)']} / I:{m['details']['Importance(중요)']} / Rel:{m['details']['Relevance(관련)']})"
                    st.markdown(f"**{score_info}**")
                    st.caption(f"📝 {m['text']}")
                    st.markdown("---")

else:
    st.info("👈 CSV 파일을 업로드해주세요.")