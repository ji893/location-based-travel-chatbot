import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os
import re
import glob
import io

# Langchain 관련 import
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# --- Streamlit 페이지 설정 및 커스텀 CSS ---
st.set_page_config(page_title="나만의 여행 플래너", layout="wide", initial_sidebar_state="expanded")

# 커스텀 CSS 정의
st.markdown(
    """
    <style>
    /* 전체 배경색 및 폰트 */
    .stApp {
        background-color: #f8f9fa; /* 밝은 회색, 거의 흰색 */
        color: #343a40; /* 어두운 회색 텍스트 */
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 제목 스타일 */
    h1 {
        color: #007bff; /* 강렬한 파란색 */
        text-align: center;
        font-size: 3.2em;
        margin-bottom: 0.6em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2 {
        color: #28a745; /* 초록색 강조 */
        font-size: 2.2em;
        border-bottom: 3px solid #e9ecef; /* 깔끔한 구분선 */
        padding-bottom: 0.4em;
        margin-top: 2em;
        margin-bottom: 1.5em;
        display: flex;
        align-items: center;
    }
    h2 .icon {
        font-size: 1.2em;
        margin-right: 10px;
    }
    h3 {
        color: #6c757d; /* 중간 회색 */
        font-size: 1.6em;
        margin-top: 1.5em;
        margin-bottom: 1em;
    }
    h4 {
        color: #495057;
        font-size: 1.2em;
        margin-top: 1em;
        margin-bottom: 0.6em;
    }

    /* 사이드바 스타일 */
    .stSidebar {
        background-color: #ffffff; /* 흰색 사이드바 */
        color: #343a40;
        border-right: 1px solid #dee2e6;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05);
    }
    .stSidebar .stButton>button {
        width: 100%;
        margin-bottom: 8px;
        border-radius: 8px;
        border: none;
        background-color: #e9ecef; /* 버튼 배경색 */
        color: #343a40;
        font-size: 1em;
        padding: 10px 15px;
        transition: all 0.2s ease-in-out;
    }
    .stSidebar .stButton>button:hover {
        background-color: #007bff; /* 호버시 색상 */
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSidebar .stSubheader {
        color: #007bff;
        text-align: center;
        margin-bottom: 1.5em;
        font-size: 1.4em;
    }
    .stSidebar .stInfo {
        background-color: #e0f7fa;
        border-left: 5px solid #00acc1;
        padding: 10px;
        border-radius: 8px;
        margin-top: 15px;
    }

    /* 입력 위젯 스타일 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div, .stMultiSelect>div>div>div {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 12px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        background-color: #ffffff;
        font-size: 1.05em;
    }
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 12px;
        background-color: #ffffff;
        font-size: 1.05em;
    }
    .stForm {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }
    .stForm button {
        margin-top: 20px;
    }

    /* 버튼 스타일 */
    .stButton>button {
        background-color: #007bff; /* 주 버튼 파란색 */
        color: white;
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        box-shadow: 0 5px 10px rgba(0,123,255,0.2);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* 호버 시 더 진하게 */
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,123,255,0.3);
    }
    /* 특정 버튼 (새로운 대화 시작하기) 스타일 */
    .stButton button[kind="secondary"] { /* Streamlit 1.28+에서 button key에 따라 style 주는 방법 */
        background-color: #6c757d; /* 회색 버튼 */
        box-shadow: 0 3px 6px rgba(108,117,125,0.2);
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #5a6268;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(108,117,125,0.3);
    }


    /* 경고/성공/정보 메시지 스타일 */
    .stAlert {
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 1.05em;
    }
    .stAlert.success {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .stAlert.warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffc107;
    }
    .stAlert.error {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .stAlert.info {
        background-color: #e0f7fa;
        color: #0056b3;
        border-left: 5px solid #007bff;
    }
    
    /* 스피너 스타일 */
    .stSpinner > div > div {
        color: #007bff; /* 스피너 색상 변경 */
    }

    /* 마크다운 테이블 스타일 (여행 계획표) */
    .st-ag .ag-header-cell {
        background-color: #007bff !important;
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.1em;
    }
    .st-ag .ag-cell {
        background-color: #ffffff !important;
        color: #343a40 !important;
        padding: 12px !important;
    }
    .st-ag .ag-row-even {
        background-color: #f8f9fa !important;
    }
    .st-ag .ag-row-odd {
        background-color: #ffffff !important;
    }
    .ag-root-wrapper {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* 입력 필드 레이블 */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label, .stNumberInput label {
        font-weight: bold;
        color: #495057;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }

    /* 구분선 */
    hr {
        margin-top: 3em;
        margin-bottom: 3em;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 123, 255, 0.75), rgba(0, 0, 0, 0));
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 파일 경로 정의 (상수) ---
VECTOR_DB_PATH = "faiss_tourist_attractions"

TOUR_CSV_FILES = [
    "./경기도역사관광지현황.csv",
    "./경기도자연관광지현황.csv",
    "./경기도체험관광지현황.csv",
    "./경기도테마관광지현황.csv",
    "./관광지정보현황(제공표준).csv",
    "./관광지현황.csv",
]

# --- 초기 파일 존재 여부 확인 ---
for f_path in TOUR_CSV_FILES:
    if not os.path.exists(f_path):
        st.error(f"필수 데이터 파일 '{f_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")
        st.stop()


# --- 1. 설정 및 초기화 함수 ---
def setup_environment():
    if 'OPENAI_API_KEY' in st.secrets:
        return st.secrets['OPENAI_API_KEY']
    else:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OpenAI API 키를 찾을 수 없습니다. Streamlit Cloud에서는 `secrets.toml`에 키를 설정하거나, 로컬에서는 `.env` 파일을 확인해주세요.")
        return api_key


def initialize_streamlit_app():
    st.title("🗺️ 나만의 AI 여행 플래너")
    st.markdown(
        """
        <div style="text-align: center; color: #6c757d; font-size: 1.1em; margin-top: -1em; margin-bottom: 2em;">
            당신의 취향에 맞춰 최적의 여행 코스를 제안해 드립니다.
        </div>
        """, unsafe_allow_html=True
    )

# --- 2. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_specific_tour_data(file_paths_list):
    combined_df = pd.DataFrame()
    if not file_paths_list:
        st.error("로드할 관광지 CSV 파일 경로가 지정되지 않았습니다. `TOUR_CSV_FILES`를 확인해주세요.")
        st.stop()
    for file_path in file_paths_list:
        if not os.path.exists(file_path):
            st.warning(f"'{file_path}' 파일을 찾을 수 없어 건너뜱니다.")
            continue
        current_encoding = 'cp949'
        try:
            df = pd.read_csv(file_path, encoding=current_encoding)
            df.columns = df.columns.str.strip()
            if "위도" not in df.columns or "경도" not in df.columns:
                st.warning(f"'{os.path.basename(file_path)}' 파일은 '위도', '경도' 컬럼이 없어 건너뜱니다.")
                continue
            name_col = None
            for candidate in ["관광지명", "관광정보명","관광지"]:
                if candidate in df.columns:
                    name_col = candidate
                    break
            if name_col is None:
                df["관광지명"] = "이름 없음"
            else:
                df["관광지명"] = df[name_col]
            address_col = None
            for candidate in ["정제도로명주소","정제지번주소","소재지도로명주소","소재지지번주소","관광지소재지지번주소","관광지소재지도로명주소"]:
                if candidate in df.columns:
                    address_col = candidate
                    break
            if address_col is None:
                df["소재지도로명주소"] = "주소 없음"
            else:
                df["소재지도로명주소"] = df[address_col]
            df = df[["위도", "경도", "관광지명", "소재지도로명주소"]]
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 처리 중 오류 발생: {e}")
    if combined_df.empty:
        st.error("지정된 파일들에서 유효한 관광지 데이터를 불러오지 못했습니다. `TOUR_CSV_FILES`와 파일 내용을 확인해주세요.")
        st.stop()
    return combined_df


# --- 벡터스토어 로딩 및 캐싱 ---
@st.cache_resource
def load_and_create_vectorstore_from_specific_files(tour_csv_files_list):
    all_city_tour_docs = []
    for file_path in tour_csv_files_list:
        if not os.path.exists(file_path):
            st.warning(f"벡터스토어 생성을 위해 '{file_path}' 파일을 찾을 수 없어 건너뜱니다.")
            continue
        current_encoding = 'cp949'
        try:
            city_tour_loader = CSVLoader(file_path=file_path, encoding=current_encoding, csv_args={'delimiter': ','})
            all_city_tour_docs.extend(city_tour_loader.load())
        except Exception as e:
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 로드 중 오류 발생 (벡터스토어): {e}")
    all_documents = all_city_tour_docs
    if not all_documents:
        st.error("벡터스토어를 생성할 문서가 없습니다. CSV 파일 경로와 내용을 확인해주세요.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = text_splitter.split_documents(all_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore

@st.cache_resource()
def get_vectorstore_cached(tour_csv_files_list):
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                VECTOR_DB_PATH,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"기존 벡터 DB 로딩 실패: {e}. 새로 생성합니다.")
            return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)
    else:
        return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)


# --- Haversine distance function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# --- 3. 사용자 입력 및 UI 로직 함수 ---
def get_user_inputs_ui():
    """사용자로부터 나이, 여행 스타일, 현재 위치, 그리고 추가 여행 계획 정보를 입력받는 UI를 표시합니다."""
    
    st.markdown("---")
    st.markdown("## <span class='icon'>1️⃣</span> 사용자 정보 입력", unsafe_allow_html=True)
    
    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.selectbox("나이대 선택", ["10대", "20대", "30대", "40대", "50대 이상"], key='age_selectbox')
        with col2:
            travel_style = st.multiselect("선호하는 여행 스타일 (복수 선택 가능)", ["자연", "역사", "체험", "휴식", "문화", "가족", "액티비티"], key='travel_style_multiselect')
        
        st.markdown("### 추가 여행 계획 정보")
        trip_duration_days = st.number_input("여행 기간 (일)", min_value=1, value=3, key='trip_duration')
        estimated_budget = st.number_input("예상 예산 (원, 총 금액)", min_value=0, value=500000, step=10000, key='estimated_budget')
        num_travelers = st.number_input("여행 인원 (명)", min_value=1, value=2, key='num_travelers')
        special_requests = st.text_area("특별히 고려할 사항 (선택 사항)", help="예: 유모차 사용, 고령자 동반, 특정 음식 선호 등", key='special_requests')
        
        submitted_user_info = st.form_submit_button("다음 단계로 이동 👉")

    return age, travel_style, trip_duration_days, estimated_budget, num_travelers, special_requests, submitted_user_info

def get_location_ui():
    st.markdown("---")
    st.markdown("## <span class='icon'>2️⃣</span> 현재 위치 가져오기", unsafe_allow_html=True)
    st.info("정확한 추천을 위해 위치 정보를 가져와주세요. 만약 위치 정보 사용이 어렵다면 수동으로 입력할 수 있습니다.")
    
    location = streamlit_geolocation()

    user_lat_final, user_lon_final = None, None

    if location and "latitude" in location and "longitude" in location:
        temp_lat = location.get("latitude")
        temp_lon = location.get("longitude")
        if temp_lat is not None and temp_lon is not None:
            user_lat_final = temp_lat
            user_lon_final = temp_lon
            st.success(f"✅ 현재 위치: 위도 {user_lat_final:.7f}, 경도 {user_lon_final:.7f}")
        else:
            st.warning("⚠️ 위치 정보를 불러오지 못했습니다. 수동으로 입력해 주세요.")
    else:
        st.warning("⚠️ 위치 정보를 사용할 수 없습니다. 수동으로 위도, 경도를 입력해 주세요.")

    with st.expander("직접 위치 입력하기 (선택 사항)", expanded=(user_lat_final is None or user_lon_final is None)):
        default_lat = st.session_state.get("user_lat", 37.5665) # 서울 시청 기본 위도
        default_lon = st.session_state.get("user_lon", 126.9780) # 서울 시청 기본 경도

        manual_lat = st.number_input("위도", value=float(default_lat), format="%.7f", key="manual_lat_input")
        manual_lon = st.number_input("경도", value=float(default_lon), format="%.7f", key="manual_lon_input")

        if manual_lat != 0.0 or manual_lon != 0.0:
            user_lat_final = manual_lat
            user_lon_final = manual_lon
            st.info(f"수동 설정된 위치: 위도 {user_lat_final:.7f}, 경도 {user_lon_final:.7f}")
        else:
            if user_lat_final is None or user_lon_final is None: # 자동 위치 실패 시에만 에러
                st.error("❌ 유효한 위도 및 경도 값이 입력되지 않았습니다. 0이 아닌 값을 입력해주세요.")

    st.session_state.user_lat = user_lat_final
    st.session_state.user_lon = user_lon_final
    
    st.markdown("---")
    submitted_location = st.button("위치 정보 확정 및 다음 단계로 이동 👉", key="submit_location_button")
    
    return user_lat_final, user_lon_final, submitted_location

def get_query_ui(current_input):
    st.markdown("## <span class='icon'>3️⃣</span> 무엇을 도와드릴까요?", unsafe_allow_html=True)
    user_query = st.text_input(
        "어떤 여행을 계획하고 계신가요? (예: 가족과 함께 즐길 수 있는 자연 테마 여행)", 
        value=current_input, 
        key="user_input",
        placeholder="예: 서울 근교에서 가을에 방문하기 좋은 역사적인 장소를 추천해주세요."
    )
    st.markdown("---")
    submitted_query = st.button("🚀 여행 계획 추천받기", use_container_width=True, key="submit_query_button")
    return user_query, submitted_query

# --- 4. 추천 로직 함수 (Langchain API 변경: create_retrieval_chain 사용) (프롬프트 수정) ---
@st.cache_resource
def get_qa_chain(_vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    qa_prompt = PromptTemplate.from_template(
        """
당신은 사용자 위치 기반 여행지 추천 및 상세 여행 계획 수립 챗봇입니다.
사용자의 나이대, 여행 성향, 현재 위치 정보, 그리고 다음의 추가 정보를 참고하여 사용자가 입력한 질문에 가장 적합한 관광지를 추천하고, 이를 바탕으로 상세한 여행 계획을 수립해 주세요.
**관광지 추천 시 사용자 위치로부터의 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 답변에서 거리를 직접 언급하지 마십시오.**
특히, 사용자의 현재 위치({user_lat}, {user_lon})에서 가까운 장소들을 우선적으로 고려하여 추천하고 계획을 세워주세요.
꼭꼭 사용자 현재 위치와 가까운 곳을 최우선으로 해주고 사용자가 선택한 성향에 맞게 추천해주세요.

[관광지 데이터]
{context}

[사용자 정보]
나이대: {age}
여행 성향: {travel_style}
현재 위치 (위도, 경도): {user_lat}, {user_lon}
여행 기간: {trip_duration_days}일
예상 예산: {estimated_budget}원
여행 인원: {num_travelers}명
특별 고려사항: {special_requests}

[사용자 질문]
{input}

다음 지침에 따라 상세한 여행 계획을 세워주세요:
1.  **관광지 추천:** 질문에 부합하고, 사용자 위치에서 가까운 1~3개의 주요 관광지를 추천하고, 각 관광지에 대한 다음 정보를 제공하세요.
    * 관광지 이름: [관광지명]
    * 주소: [주소]
    * 주요 시설/특징: [정보]
    **[참고: 사용자 위치 기준 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 이 항목은 제외합니다.]**
    
2.  **추천된 관광지를 포함하여, 사용자 정보와 질문에 기반한 {trip_duration_days}일간의 상세 여행 계획을 일자별로 구성해 주세요.**
    * 각 날짜별로 방문할 장소(식당, 카페, 기타 활동 포함), 예상 시간, 간단한 활동 내용을 포함하세요.
    * 예산을 고려하여 적절한 식사 장소나 활동을 제안할 수 있습니다.
    * 이동 경로(예: "도보 15분", "버스 30분")를 간략하게 언급해 주세요.
    * 계획은 명확하고 이해하기 쉽게 작성되어야 합니다.

[답변 예시]
**추천 관광지:**
- 관광지 이름: [관광지명 1]
    - 주소: [주소 1]
    - 주요 시설/특징: [정보 1]
- 관광지 이름: [관광지명 2]
    - 주소: [주소 2]
    - 주요 시설/특징: [정보 2]

**상세 여행 계획 ({trip_duration_days}일):**
다음 표 형식으로 일자별 상세 계획을 작성해 주세요. 컬럼명은 '일차', '시간', '활동', '예상 장소', '이동 방법'으로 해주세요.
| 일차 | 시간 | 활동 | 예상 장소 | 이동 방법 |
|---|---|---|---|---|
| 1일차 | 오전 (9:00 - 12:00) | [활동 내용] | [장소명] | [이동 방법] |
| 1일차 | 점심 (12:00 - 13:00) | [식사] | [식당명] | - |
| 1일차 | 오후 (13:00 - 17:00) | [활동 내용] | [장소명] | [이동 방법] |
| 1일차 | 저녁 (17:00 이후) | [활동 내용] | [장소명 또는 자유 시간] | - |
| 2일차 | ... | ... | ... | ... |
**중요: '일차' 컬럼의 경우, 같은 일차의 여러 활동이 있을 경우 첫 번째 활동에만 해당 '일차'를 명시하고, 나머지 활동 행의 '일차' 셀은 비워두세요 (예: "| | 시간 | 활동 | 예상 장소 | 이동 방법 |"). 이렇게 해야 표에서 '일차'가 자동으로 병합되어 보입니다.**
"""
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 15})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# --- 5. 메인 앱 실행 로직 ---
if __name__ == "__main__":
    openai_api_key = setup_environment()
    if not openai_api_key:
        st.stop()

    initialize_streamlit_app()

    vectorstore = get_vectorstore_cached(TOUR_CSV_FILES)
    tour_data_df = load_specific_tour_data(TOUR_CSV_FILES)

    # 세션 상태 초기화: current_step 추가
    if "current_step" not in st.session_state:
        st.session_state.current_step = "user_info" # 초기 단계 설정: 사용자 정보 입력
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    if "selected_conversation_index" not in st.session_state:
        st.session_state.selected_conversation_index = None

    # 사이드바에 이전 대화 목록 표시
    with st.sidebar:
        st.subheader("💡 이전 대화")
        if st.session_state.conversations:
            for i, conv in enumerate(reversed(st.session_state.conversations)):
                original_index = len(st.session_state.conversations) - 1 - i
                
                preview_text = f"Q: {conv['user_query'][:20]}" + ('...' if len(conv['user_query']) > 20 else '')
                
                if st.button(preview_text, key=f"sidebar_conv_{original_index}"):
                    st.session_state.selected_conversation_index = original_index
                    st.session_state.current_step = "show_history" # 기록 보기 단계로 전환
                    st.rerun()
            st.markdown("---")
            if st.button("새로운 여행 계획 시작", key="new_plan_sidebar_button"):
                st.session_state.selected_conversation_index = None
                st.session_state.current_step = "user_info" # 새로운 계획 시작 시 사용자 정보 단계로
                st.session_state.current_input = ""
                st.rerun()

        else:
            st.info("이전 대화가 없습니다.")
    
    # --- 메인 콘텐츠 영역 (단계별 UI) ---

    # 0. 이전 대화 기록 보기
    if st.session_state.selected_conversation_index is not None:
        selected_conv = st.session_state.conversations[st.session_state.selected_conversation_index]
        
        st.markdown("---")
        st.markdown(f"## 💬 이전 대화 내용 ({st.session_state.selected_conversation_index + 1}번째 대화)", unsafe_allow_html=True)
        st.markdown("### 질문:")
        st.info(selected_conv['user_query'])
        
        if 'travel_style_selected' in selected_conv and selected_conv['travel_style_selected'] and selected_conv['travel_style_selected'] != '특정 없음':
            st.markdown("### 선택된 여행 성향:")
            st.markdown(f"**{selected_conv['travel_style_selected']}**")

        st.markdown("### 답변:")
        st.markdown(selected_conv['chatbot_response'])
        
        st.markdown("---")
        if st.button("✨ 새로운 여행 계획 시작하기", key="start_new_conv_button", type="primary"):
            st.session_state.selected_conversation_index = None
            st.session_state.current_step = "user_info"
            st.session_state.current_input = ""
            st.rerun()
        
    # 1. 사용자 정보 입력 단계
    elif st.session_state.current_step == "user_info":
        st.subheader("여행을 위한 기본 정보를 알려주세요.")
        age, travel_style_list, trip_duration_days, estimated_budget, num_travelers, special_requests, submitted_user_info = get_user_inputs_ui()
        
        if submitted_user_info:
            st.session_state.age = age
            st.session_state.travel_style_list = travel_style_list
            st.session_state.trip_duration_days = trip_duration_days
            st.session_state.estimated_budget = estimated_budget
            st.session_state.num_travelers = num_travelers
            st.session_state.special_requests = special_requests
            st.session_state.current_step = "get_location" # 다음 단계로 이동
            st.rerun()

    # 2. 위치 정보 입력 단계
    elif st.session_state.current_step == "get_location":
        st.subheader("현재 위치를 확인해주세요.")
        current_user_lat, current_user_lon, submitted_location = get_location_ui()

        if submitted_location:
            if current_user_lat is None or current_user_lon is None:
                st.error("위치 정보가 유효하지 않습니다. 다시 시도해주세요.")
            else:
                st.session_state.current_user_lat = current_user_lat
                st.session_state.current_user_lon = current_user_lon
                st.session_state.current_step = "get_query" # 다음 단계로 이동
                st.rerun()
        
        if st.button("👈 이전 단계로", key="back_to_user_info"):
            st.session_state.current_step = "user_info"
            st.rerun()

    # 3. 질문 입력 및 결과 출력 단계
    elif st.session_state.current_step == "get_query":
        st.subheader("이제 여행에 대해 자세히 질문해주세요.")
        user_query, submitted_query = get_query_ui(st.session_state.current_input)

        if submitted_query:
            lat_to_invoke = st.session_state.current_user_lat
            lon_to_invoke = st.session_state.current_user_lon

            age_to_invoke = st.session_state.age
            travel_style_to_invoke = ', '.join(st.session_state.travel_style_list) if st.session_state.travel_style_list else '특정 없음'
            trip_duration_days_to_invoke = st.session_state.trip_duration_days
            estimated_budget_to_invoke = st.session_state.estimated_budget
            num_travelers_to_invoke = st.session_state.num_travelers
            special_requests_to_invoke = st.session_state.special_requests

            if lat_to_invoke is None or lon_to_invoke is None:
                st.warning("⚠️ 위치 정보가 없으므로 답변을 생성할 수 없습니다. 다시 위치 정보 단계로 돌아가주세요.")
            elif not user_query.strip():
                st.warning("⚠️ 질문을 입력해주세요.")
            else:
                with st.spinner("⏳ 당신만을 위한 최적의 여행 계획을 수립 중입니다... 잠시만 기다려 주세요!"):
                    try:
                        qa_chain = get_qa_chain(vectorstore) # 동적으로 체인을 가져옴 (API 키 문제 해결 목적)
                        response = qa_chain.invoke({
                            "input": user_query,
                            "age": age_to_invoke,
                            "travel_style": travel_style_to_invoke,
                            "user_lat": lat_to_invoke,
                            "user_lon": lon_to_invoke,
                            "trip_duration_days": trip_duration_days_to_invoke,
                            "estimated_budget": estimated_budget_to_invoke,
                            "num_travelers": num_travelers_to_invoke,
                            "special_requests": special_requests_to_invoke
                        })

                        rag_result_text = response["answer"]

                        processed_output_lines = []
                        processed_place_names = set()
                        table_plan_text = ""
                        in_plan_section = False

                        for line in rag_result_text.split('\n'):
                            if "상세 여행 계획" in line and "일차 | 시간 | 활동" not in line:
                                processed_output_lines.append(line)
                                in_plan_section = True
                                continue 

                            if not in_plan_section:
                                name_match = re.search(r"관광지 이름:\s*(.+)", line)
                                if name_match:
                                    current_place_name = name_match.group(1).strip()
                                    if current_place_name not in processed_place_names:
                                        processed_output_lines.append(line)
                                        processed_place_names.add(current_place_name)

                                        found_place_data = tour_data_df[
                                            (tour_data_df['관광지명'].str.strip() == current_place_name) &
                                            (pd.notna(tour_data_df['위도'])) &
                                            (pd.notna(tour_data_df['경도']))
                                        ]
                                        if not found_place_data.empty:
                                            place_lat = found_place_data['위도'].iloc[0]
                                            place_lon = found_place_data['경도'].iloc[0]
                                            distance = haversine(lat_to_invoke, lon_to_invoke, place_lat, place_lon)
                                            processed_output_lines.append(f"- 사용자 위치 기준 거리(km): 약 {distance:.2f} km")
                                        else:
                                            processed_output_lines.append("- 사용자 위치 기준 거리(km): 정보 없음 (데이터 불일치 또는 좌표 누락)")
                                else:
                                    if not re.search(r"거리\(km\):", line):
                                        processed_output_lines.append(line)
                            else:
                                table_plan_text += line + "\n"
                        
                        st.markdown("---")
                        st.markdown("## ✨ 추천 결과 및 상세 여행 계획", unsafe_allow_html=True)
                        st.markdown("\n".join(processed_output_lines))

                        if table_plan_text.strip():
                            try:
                                plan_lines = table_plan_text.strip().split('\n')
                                
                                # 헤더와 구분선이 모두 존재하는지 확인
                                if len(plan_lines) >= 2 and plan_lines[0].count('|') >= 2 and plan_lines[1].count('|') >= 2 and all(re.match(r'^-+$', s.strip()) for s in plan_lines[1].split('|') if s.strip()):
                                    header = [h.strip() for h in plan_lines[0].split('|') if h.strip()]
                                    data_rows = []
                                    for row_str in plan_lines[2:]:
                                        if row_str.strip() and row_str.startswith('|'):
                                            # 행 파싱 시 양 끝의 빈 문자열 제거
                                            parsed_row = [d.strip() for d in row_str.split('|') if d.strip() or (d == '' and (len(parsed_row) < len(header)))]
                                            
                                            # '일차' 컬럼이 비어있을 때를 대비하여 첫 번째 요소가 빈 문자열이면 건너뛰지 않도록 수정
                                            if parsed_row and parsed_row[0] == '' and len(parsed_row) < len(header): # 빈 첫 컬럼이 있고, 아직 헤더 수보다 적을 때
                                                parsed_row = parsed_row # 그대로 유지
                                            elif parsed_row and parsed_row[0] == '' and len(parsed_row) == len(header): # 빈 첫 컬럼이 있고, 헤더 수와 일치할 때
                                                parsed_row = parsed_row[1:] # 실제 데이터 시작
                                            
                                            # 데이터 행의 길이가 헤더와 맞는지 확인
                                            if len(parsed_row) == len(header):
                                                data_rows.append(parsed_row)
                                            else:
                                                st.warning(f"⚠️ 테이블 행 데이터와 헤더 불일치: {parsed_row}. 스킵합니다.")

                                    if data_rows:
                                        temp_plan_df = pd.DataFrame(data_rows, columns=header)
                                        
                                        if '일차' in temp_plan_df.columns:
                                            # '일차' 컬럼의 중복 값을 빈 문자열로 대체하여 병합 효과 (Streamlit 테이블에서는 자동 병합되지 않으므로 시각적으로만)
                                            # 이전 값과 같으면 현재 값을 빈 문자열로 만듦
                                            current_day = ''
                                            for i in range(len(temp_plan_df)):
                                                if temp_plan_df.loc[i, '일차'] == current_day:
                                                    temp_plan_df.loc[i, '일차'] = ''
                                                else:
                                                    current_day = temp_plan_df.loc[i, '일차']

                                            st.subheader("🗓️ 추천 여행 계획표")
                                            st.dataframe(temp_plan_df, use_container_width=True) 
                                        else:
                                            st.subheader("🗓️ 추천 여행 계획표")
                                            st.dataframe(temp_plan_df, use_container_width=True)
                                            st.warning("⚠️ 여행 계획에 '일차' 컬럼이 없어 그룹화하여 표시할 수 없습니다.")
                                    else:
                                        st.warning("⚠️ 유효한 여행 계획 테이블 데이터가 없습니다.")
                                else:
                                    st.warning("⚠️ 여행 계획 테이블의 헤더 또는 구분선 형식이 올바르지 않습니다.")
                            except Exception as parse_e:
                                st.error(f"❌ 여행 계획 테이블 파싱 중 오류 발생: {parse_e}. LLM 응답 형식을 확인해주세요.")
                        else:
                            st.info("ℹ️ 상세 여행 계획이 제공되지 않았습니다.")
                        
                        st.session_state.conversations.append({
                            "user_query": user_query,
                            "chatbot_response": rag_result_text,
                            "travel_style_selected": travel_style_to_invoke
                        })
                        st.session_state.current_input = "" # 입력창 초기화
                        # 결과 표시 후 새로운 대화 시작 버튼을 누르도록 유도
                        st.markdown("---")
                        if st.button("새로운 여행 계획 시작하기", key="post_result_new_conv", type="secondary"):
                            st.session_state.current_step = "user_info"
                            st.rerun()

                    except ValueError as ve:
                        st.error(f"❌ 체인 호출 중 오류 발생: {ve}. 입력 키를 확인해주세요.")
                    except Exception as e:
                        st.error(f"❌ 예상치 못한 오류 발생: {e}")
        
        # 이전 단계로 돌아가는 버튼
        if st.button("👈 이전 단계로 (위치 정보)", key="back_to_location"):
            st.session_state.current_step = "get_location" # 'location' 단계로 돌아가도록 수정
            st.rerun()
