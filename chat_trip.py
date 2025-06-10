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

# .env 파일 로드 (로컬 개발 시 사용. Streamlit Cloud에서는 Secrets 사용 권장)
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="🚂 관광지 추천 챗봇", layout="wide")

# --- 커스텀 CSS 정의 ---
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
    /* Streamlit 1.28+에서 button key에 따라 style 주는 방법 */
    .stButton button[data-testid="stFormSubmitButton"] {
        background-color: #28a745; /* 예: 제출 버튼은 초록색 */
        box-shadow: 0 5px 10px rgba(40,167,69,0.2);
    }
    .stButton button[data-testid="stFormSubmitButton"]:hover {
        background-color: #218838;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(40,167,69,0.3);
    }
    
    /* 일반 버튼 중 secondary type */
    .stButton button[kind="secondary"] { 
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
    /* Streamlit의 st.dataframe은 AgGrid 기반이므로 AgGrid 관련 클래스를 사용합니다. */
    /* 아래 CSS는 st.dataframe에만 적용될 수 있습니다. st.markdown으로 생성된 테이블에는 적용되지 않을 수 있습니다. */
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
# GitHub 저장소에 업로드할 때 이 경로가 올바르게 설정되어 있어야 합니다.
VECTOR_DB_PATH = "faiss_tourist_attractions"

# 로드할 개별 관광지 CSV 파일 목록을 직접 지정합니다.
# 이 파일들은 GitHub 저장소의 앱 스크립트와 동일한 위치 또는 지정된 상대 경로에 있어야 합니다.
TOUR_CSV_FILES = [
    "./경기도역사관광지현황.csv",
    "./경기도자연관광지현황.csv",
    "./경기도체험관광지현황.csv",
    "./경기도테마관광지현황.csv",
    "./관광지정보현황(제공표준).csv",
    "./관광지현황.csv",
    # 필요에 따라 다른 CSV 파일들을 여기에 추가하세요.
]

# --- 초기 파일 존재 여부 확인 ---
# 앱 시작 전 필수 데이터 파일의 존재 여부를 확인합니다.
for f_path in TOUR_CSV_FILES:  # 필수 관광지 CSV 파일 존재 여부 확인
    if not os.path.exists(f_path):  # 파일 존재하지 않으면
        st.error(f"필수 데이터 파일 '{f_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")  # 에러 메시지 출력
        st.stop()  # 파일이 없으면 앱 실행 중지


# --- 1. 설정 및 초기화 함수 ---
def setup_environment():  # 환경 설정 함수
    """
    환경 변수 또는 Streamlit secrets에서 OpenAI API 키를 로드합니다.
    Streamlit Cloud 환경에서는 st.secrets를 우선적으로 사용합니다.
    로컬 환경에서는 .env 파일을 로드하거나 시스템 환경 변수에서 가져옵니다.
    """
    if 'OPENAI_API_KEY' in st.secrets:  # secrets에 키가 있으면
        return st.secrets['OPENAI_API_KEY']  # 그 키 반환
    else:
        api_key = os.getenv("OPENAI_API_KEY")  # 환경변수에서 키 불러오기
        if not api_key:  # 키 없으면 에러 출력
            st.error("❌ OpenAI API 키를 찾을 수 없습니다. Streamlit Cloud에서는 `secrets.toml`에 키를 설정하거나, 로컬에서는 `.env` 파일을 확인해주세요.")
        return api_key  # 키 반환 또는 None


# --- 2. 데이터 로드 및 전처리 함수 ---

@st.cache_data  # 캐시 사용
def load_specific_tour_data(file_paths_list):  # 관광지 CSV 파일들 로드 함수
    """
    지정된 CSV 파일 목록을 로드하고, 모든 파일에 CP949 인코딩을 적용하여 병합합니다.
    '위도', '경도', '관광지명', '소재지도로명주소' 컬럼을 필수로 추출합니다.
    """
    combined_df = pd.DataFrame()  # 빈 데이터프레임 생성

    if not file_paths_list:  # 파일 리스트 없을 경우
        st.error("로드할 관광지 CSV 파일 경로가 지정되지 않았습니다. `TOUR_CSV_FILES`를 확인해주세요.")
        st.stop()

    for file_path in file_paths_list:  # 각 파일에 대해
        if not os.path.exists(file_path):  # 파일이 없으면
            st.warning(f"'{file_path}' 파일을 찾을 수 없어 건너뜁니다. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")
            continue

        current_encoding = 'cp949'  # CP949 인코딩 사용

        try:
            df = pd.read_csv(file_path, encoding=current_encoding)  # CSV 파일 읽기
            df.columns = df.columns.str.strip()  # 컬럼명 공백 제거

            if "위도" not in df.columns or "경도" not in df.columns:  # 필수 컬럼 확인
                st.warning(f"'{os.path.basename(file_path)}' 파일은 '위도', '경도' 컬럼이 없어 건너뜁니다.")
                continue

            name_col = None
            for candidate in ["관광지명", "관광정보명", "관광지"]:
                if candidate in df.columns:
                    name_col = candidate
                    break
            df["관광지명"] = df[name_col] if name_col else "이름 없음"

            address_col = None
            for candidate in ["정제도로명주소", "정제지번주소", "소재지도로명주소", "소재지지번주소", "관광지소재지지번주소", "관광지소재지도로명주소"]:
                if candidate in df.columns:
                    address_col = candidate
                    break
            df["소재지도로명주소"] = df[address_col] if address_col else "주소 없음"

            combined_df = pd.concat([combined_df, df[["위도", "경도", "관광지명", "소재지도로명주소"]]], ignore_index=True)

        except Exception as e:
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 처리 중 오류 발생: {e}")

    if combined_df.empty:
        st.error("지정된 파일들에서 유효한 관광지 데이터를 불러오지 못했습니다. `TOUR_CSV_FILES`와 파일 내용을 확인해주세요.")
        st.stop()

    combined_df.dropna(subset=['위도', '경도'], inplace=True)
    return combined_df


# --- 벡터스토어 로딩 및 캐싱 ---
@st.cache_resource  # 벡터스토어 생성을 캐시 (Streamlit 재실행 시 재사용)
def load_and_create_vectorstore_from_specific_files(tour_csv_files_list):  # 지정된 CSV 파일로부터 벡터스토어 생성
    """지정된 CSV 파일 목록을 사용하여 벡터스토어를 생성합니다."""
    all_city_tour_docs = []  # 모든 관광지 문서를 저장할 리스트

    for file_path in tour_csv_files_list:  # 각 파일 경로에 대해
        if not os.path.exists(file_path):  # 파일이 존재하지 않으면
            st.warning(f"벡터스토어 생성을 위해 '{file_path}' 파일을 찾을 수 없어 건너뜁니다.")  # 경고 출력
            continue

        current_encoding = 'cp949'  # CP949 인코딩 사용

        try:
            city_tour_loader = CSVLoader(file_path=file_path, encoding=current_encoding, csv_args={'delimiter': ','})  # CSV 로더 생성
            all_city_tour_docs.extend(city_tour_loader.load())  # 문서 로드 및 추가
        except Exception as e:  # 예외 발생 시
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 로드 중 오류 발생 (벡터스토어): {e}")  # 경고 출력

    all_documents = all_city_tour_docs  # 전체 문서 저장

    if not all_documents:  # 문서가 비어 있으면
        st.error("벡터스토어를 생성할 문서가 없습니다. CSV 파일 경로와 내용을 확인해주세요.")  # 에러 출력 후 중단
        st.stop()

    # 텍스트 분할 및 임베딩 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)  # 텍스트 분할기 설정
    docs = text_splitter.split_documents(all_documents)  # 문서 분할
    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 사용
    vectorstore = FAISS.from_documents(docs, embeddings)  # FAISS 벡터스토어 생성
    vectorstore.save_local(VECTOR_DB_PATH)  # 로컬에 벡터스토어 저장
    return vectorstore  # 벡터스토어 반환

@st.cache_resource()  # 벡터스토어 캐시 로딩 함수
def get_vectorstore_cached(tour_csv_files_list):  # 벡터스토어를 캐시에서 불러오거나 새로 생성
    """캐시된 벡터스토어를 로드하거나 새로 생성합니다."""
    if os.path.exists(VECTOR_DB_PATH):  # 기존 벡터 DB가 존재하면
        try:
            return FAISS.load_local(  # 벡터 DB 로드 시도
                VECTOR_DB_PATH,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True  # 최신 LangChain 버전에서 필요
            )
        except Exception as e:  # 로딩 실패 시
            st.warning(f"기존 벡터 DB 로딩 실패: {e}. 새로 생성합니다.")  # 경고 출력
            return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)  # 새로 생성
    else:
        return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)  # 존재하지 않으면 새로 생성

# --- Haversine 거리 계산 함수 ---
def haversine(lat1, lon1, lat2, lon2):  # 두 지점 간 거리 계산 함수
    """두 위도/경도 지점 간의 거리를 킬로미터 단위로 계산합니다 (하버사인 공식)."""
    R = 6371  # 지구 반지름 (킬로미터)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  # 각도를 라디안으로 변환
    dlon = lon2 - lon1  # 경도 차이
    dlat = lat2 - lat1  # 위도 차이
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2  # 하버사인 공식 부분
    c = 2 * atan2(sqrt(a), sqrt(1 - a))  # 중심각 계산
    distance = R * c  # 거리 계산
    return distance  # 결과 반환


# --- 3. 사용자 입력 및 UI 로직 함수 ---
def get_user_inputs_ui():  # 사용자 입력 UI 구성 함수
    """사용자로부터 나이, 여행 스타일, 현재 위치, 그리고 추가 여행 계획 정보를 입력받는 UI를 표시합니다."""
    col1, col2, col3 = st.columns([1, 2, 1])  # 3개 컬럼 레이아웃 생성
    with col2:  # 가운데 컬럼에 입력 요소 배치
        st.markdown("#### 🧑‍💻 사용자 정보 입력")  # 제목 마크다운
        age = st.selectbox("나이대 선택", ["10대", "20대", "30대", "40대", "50대 이상"], key='age_selectbox')  # 나이대 선택박스
        travel_style = st.multiselect("여행 스타일", ["자연", "역사", "체험", "휴식", "문화", "가족", "액티비티"], key='travel_style_multiselect')  # 여행 스타일 다중 선택

    st.header("① 위치 가져오기")  # 위치 가져오기 섹션 제목
    location = streamlit_geolocation()  # 위치 정보 요청 (streamlit_geolocation 호출)

    user_lat_final, user_lon_final = None, None  # 최종 위도, 경도 초기화

    if location and "latitude" in location and "longitude" in location:  # 위치 정보 유효 체크
        temp_lat = location.get("latitude")  # 위도 임시 저장
        temp_lon = location.get("longitude")  # 경도 임시 저장
        if temp_lat is not None and temp_lon is not None:  # 위/경도 존재 시
            user_lat_final = temp_lat  # 최종 위도 저장
            user_lon_final = temp_lon  # 최종 경도 저장
            st.success(f"📍 현재 위치: 위도 **{user_lat_final:.7f}**, 경도 **{user_lon_final:.7f}**")  # 위치 정보 성공 메시지
        else:
            st.warning("📍 위치 정보를 불러오지 못했습니다. 수동으로 입력해 주세요.")  # 위치 정보 없음 경고
    else:
        st.warning("위치 정보를 사용할 수 없습니다. 수동으로 위도, 경도를 입력해 주세요.")  # 위치 정보 자체가 없을 경우 경고

    # 위치 정보가 없거나 0,0인 경우 수동 입력 UI 표시
    if user_lat_final is None or user_lon_final is None or (user_lat_final == 0.0 and user_lon_final == 0.0):
        # 세션 상태에서 기본값 가져오기 (이전 입력 유지)
        default_lat = st.session_state.get("user_lat", 37.5665)  # 기본 위도: 서울 시청
        default_lon = st.session_state.get("user_lon", 126.9780)  # 기본 경도: 서울 시청

        st.subheader("직접 위치 입력 (선택 사항)")  # 수동 위치 입력 서브타이틀
        manual_lat = st.number_input("위도", value=float(default_lat), format="%.7f", key="manual_lat_input")  # 위도 입력란
        manual_lon = st.number_input("경도", value=float(default_lon), format="%.7f", key="manual_lon_input")  # 경도 입력란

        # 수동 입력값이 유효하면 최종 값으로 설정
        if manual_lat != 0.0 or manual_lon != 0.0:
            user_lat_final = manual_lat  # 최종 위도 재설정
            user_lon_final = manual_lon  # 최종 경도 재설정
            st.info(f"수동 입력 위치: 위도 **{user_lat_final:.7f}**, 경도 **{user_lon_final:.7f}**")  # 수동 입력 안내
        else:
            st.error("유효한 위도 및 경도 값이 입력되지 않았습니다. 0이 아닌 값을 입력해주세요.")  # 유효하지 않은 입력 에러
            user_lat_final = None  # None으로 유지
            user_lon_final = None

    # 최종 결정된 위도, 경도를 세션 상태에 저장
    st.session_state.user_lat = user_lat_final
    st.session_state.user_lon = user_lon_final

    st.markdown("#### 📆 추가 여행 계획 정보")  # 추가 여행 계획 섹션 제목
    trip_duration_days = st.number_input("여행 기간 (일)", min_value=1, value=3, key='trip_duration')  # 여행 기간 입력
    estimated_budget = st.number_input("예상 예산 (원, 총 금액)", min_value=0, value=500000, step=10000, key='estimated_budget')  # 예상 예산 입력
    num_travelers = st.number_input("여행 인원 (명)", min_value=1, value=2, key='num_travelers')  # 여행 인원 입력
    special_requests = st.text_area("특별히 고려할 사항 (선택 사항)", help="예: 유모차 사용, 고령자 동반, 특정 음식 선호 등", key='special_requests')  # 특이사항 입력란

    return age, travel_style, user_lat_final, user_lon_final, trip_duration_days, estimated_budget, num_travelers, special_requests  # 입력값 반환

# --- 4. 추천 로직 함수 (Langchain API 변경: create_retrieval_chain 사용) (프롬프트 수정) ---
@st.cache_resource  # 추천 체인 캐시
def get_qa_chain(_vectorstore):  # QA 체인 생성 함수
    """
    LangChain을 사용하여 질문-답변 체인(QA Chain)을 생성하고 캐시합니다.
    이 체인은 사용자의 입력과 검색된 문서를 기반으로 답변을 생성합니다.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)  # GPT-4o LLM 인스턴스 생성 (온도 0.7)

    qa_prompt = PromptTemplate.from_template(
        """
당신은 사용자 위치 기반 여행지 추천 및 상세 여행 계획 수립 챗봇입니다.
사용자의 나이대, 여행 성향, 현재 위치 정보, 그리고 다음의 추가 정보를 참고하여 사용자가 입력한 질문에 가장 적합한 관광지를 추천하고, 이를 바탕으로 상세한 여행 계획을 수립해 주세요.
**관광지 추천 시 사용자 위치로부터의 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 답변에서 거리를 직접 언급하지 마십시오.**
특히, 사용자의 현재 위치({user_lat}, {user_lon})에서 가까운 장소들을 우선적으로 고려하여 추천하고 사용자가 선택한 성향에 맞게 추천해주세요.

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
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 15})  # 상위 15개 문서 검색
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# --- 5. 메인 앱 실행 로직 ---
if __name__ == "__main__":
    # OpenAI API 키 설정
    openai_api_key = setup_environment()
    if not openai_api_key:
        st.stop()  # API 키가 없으면 앱 실행 중지
    
    # 세션 상태 초기화: 'app_started' 플래그 추가
    if "app_started" not in st.session_state:
        st.session_state.app_started = False
    
    # 세션 상태 초기화 및 이전 대화 기록 관리
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
        st.session_state.current_input = ""
        st.session_state.selected_conversation_index = None
    
    # 이전 messages 상태가 남아있을 경우 삭제 (LangChain 챗봇 메시지와 혼동 방지)
    if "messages" in st.session_state:
        del st.session_state.messages

    # 시작 화면 (앱이 아직 시작되지 않았을 때만 표시)
    if not st.session_state.app_started:
        st.title("🚂 떠나자! 맞춤형 여행 계획 챗봇")
        st.markdown("### 당신의 완벽한 여행을 위한 AI 파트너")
        
        # 이미지 파일 경로 (예: train.jpg)
        local_image_path = "./train.jpg" 
        
        # 이미지 파일 존재 여부 확인 및 표시
        if os.path.exists(local_image_path):
            st.image(local_image_path, 
                     caption="여행의 시작은 지금부터!", 
                     use_container_width=True) 
        else:
            st.warning(f"시작 화면 이미지를 찾을 수 없습니다: {local_image_path}")
            # 대체 이미지 URL (필요 시 주석 해제 가능)
            # st.image("https://images.unsplash.com/photo-1542171124-ed989b5c3ee5?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
            #           caption="여행의 시작은 비행기에서부터!", 
            #           use_container_width=True)

        st.write("""
        이 챗봇은 당신의  여행 스타일, 현재 위치를 기반으로 최적의 관광지를 추천하고, 
        상세한 일자별 여행 계획을 세워줍니다. 
        이제 번거로운 계획은 AI에게 맡기고 즐거운 여행만 준비하세요!
        """)
        
        if st.button("🚂 여행 계획 시작하기"):
            st.session_state.app_started = True
            st.rerun()  # 앱 다시 시작하여 챗봇 화면으로 전환

    else:  # 앱 시작 플래그가 True인 경우 챗봇 화면 표시
        st.title("🗺️ 위치 기반 관광지 추천 및 여행 계획 챗봇")
        
        # 벡터스토어 및 관광지 데이터 로드
        vectorstore = get_vectorstore_cached(TOUR_CSV_FILES)
        tour_data_df = load_specific_tour_data(TOUR_CSV_FILES)
        qa_chain = get_qa_chain(vectorstore)

        # 사용자 입력 UI 호출
        (
            age, travel_style, user_lat, user_lon, 
            trip_duration, budget, num_travelers, special_requests
        ) = get_user_inputs_ui()

        # 입력값이 모두 유효할 때만 질문 입력 UI 표시 및 답변 생성
        if user_lat is not None and user_lon is not None:
            user_question = st.text_input("여행에 대해 궁금한 점을 입력하세요.", key="user_question_input")
            
            if st.button("질문하기"):
                if user_question.strip() == "":
                    st.warning("질문 내용을 입력해 주세요.")
                else:
                    with st.spinner("AI가 여행 계획을 분석 중입니다..."):
                        # 예: 추가 컨텍스트를 프롬프트에 포함할 수도 있습니다.
                        context = f"나이대: {age}, 여행 스타일: {', '.join(travel_style)}, 위치: ({user_lat:.7f}, {user_lon:.7f}), 여행 기간: {trip_duration}일, 예산: {budget}원, 인원: {num_travelers}명, 특이사항: {special_requests}"
                        query = f"{context}\n\n사용자 질문: {user_question}"

                        # QA 체인 실행
                        try:
                            # LangChain의 invoke 메서드는 딕셔너리 형태로 인풋을 받습니다.
                            response = qa_chain.invoke({"input": user_question, 
                                                         "age": age, 
                                                         "travel_style": ', '.join(travel_style), 
                                                         "user_lat": user_lat, 
                                                         "user_lon": user_lon, 
                                                         "trip_duration_days": trip_duration, 
                                                         "estimated_budget": budget, 
                                                         "num_travelers": num_travelers, 
                                                         "special_requests": special_requests})
                            answer = response['answer']
                        except Exception as e:
                            st.error(f"AI 응답 생성 중 오류가 발생했습니다: {e}")
                            answer = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."
                            
                        # 대화 기록에 추가
                        st.session_state.conversations.append({
                            "question": user_question,
                            "answer": answer,
                            "user_lat": user_lat, # 현재 대화의 위도 저장
                            "user_lon": user_lon, # 현재 대화의 경도 저장
                            "travel_style_selected": ', '.join(travel_style) if travel_style else '특정 없음', # 선택된 여행 스타일 저장
                            "trip_duration": trip_duration,
                            "budget": budget,
                            "num_travelers": num_travelers,
                            "special_requests": special_requests
                        })
                        # 새로 생성된 답변이 가장 최근 것이므로 자동으로 선택
                        st.session_state.selected_conversation_index = len(st.session_state.conversations) - 1
                        st.rerun() # 화면 갱신하여 새 답변 표시

        # 사이드바: 이전 대화 기록 관리
        with st.sidebar:
            st.subheader("💡 이전 대화")
            if st.session_state.conversations:
                # 최신 대화를 먼저 보여주기 위해 역순으로 반복
                for i, conv in enumerate(reversed(st.session_state.conversations)):
                    original_index = len(st.session_state.conversations) - 1 - i
                    
                    # 대화 미리보기 텍스트 생성
                    preview_text = ""
                    if 'travel_style_selected' in conv and conv['travel_style_selected'] and conv['travel_style_selected'] != '특정 없음':
                        preview_text += f"성향: {conv['travel_style_selected']}"
                    
                    # 질문 내용 추가, 너무 길면 잘라냄
                    # 'user_query' 대신 'question' 사용
                    if 'question' in conv and conv['question'].strip():
                        query_preview = conv['question'].strip()
                        if len(query_preview) > 20:
                            query_preview = query_preview[:17] + '...'
                        if preview_text: # 성향 정보가 있다면 줄바꿈
                            preview_text += f"\n질문: {query_preview}"
                        else:
                            preview_text = f"질문: {query_preview}"
                    
                    if not preview_text: # 둘 다 없으면 기본 텍스트
                        preview_text = f"대화 {original_index + 1}"
                        
                    # 버튼 생성
                    if st.button(preview_text, key=f"sidebar_conv_{original_index}"):
                        st.session_state.selected_conversation_index = original_index
                        st.rerun() # 선택된 대화로 화면 업데이트
            else:
                st.info("이전 대화가 없습니다.")
            
            # 새로운 대화 시작 버튼 (사이드바에 배치)
            if st.button("✨ 새로운 대화 시작하기", key="new_conversation_sidebar_button"):
                st.session_state.selected_conversation_index = None
                st.session_state.conversations = [] # 모든 대화 기록 초기화
                st.session_state.current_input = ""
                st.rerun()

# --- 메인 콘텐츠 영역: 선택된 대화 표시 ---
# 이 부분은 'else: # 앱 시작 플래그가 True인 경우 챗봇 화면 표시' 블록 안에 있어야 합니다.
# 사용자가 질문하기 버튼을 눌렀을 때만 이전 대화가 생성되도록 로직을 분리했습니다.
if st.session_state.selected_conversation_index is not None:
    st.header("📝 이전 대화 내용")
    
    # 선택된 대화 객체를 세션 상태에서 가져오기
    selected_conv = st.session_state.conversations[st.session_state.selected_conversation_index]
    
    # 사용자 질문 부분 표시
    st.subheader("🗣️ 질문:")
    st.markdown(f"**{selected_conv['question']}**") # 'user_query' 대신 'question' 사용
    
    # 선택된 여행 스타일이 있을 경우 별도 표시 (특정 없음이 아닌 경우만)
    if 'travel_style_selected' in selected_conv and selected_conv['travel_style_selected'] and selected_conv['travel_style_selected'] != '특정 없음':
        st.subheader("🌟 선택된 여행 스타일:")
        st.markdown(selected_conv['travel_style_selected'])

    # 챗봇 답변 표시 영역 시작
    st.subheader("💡 답변:")
    
    # LLM이 생성한 응답 텍스트 가져오기
    # 'chatbot_response' 대신 'answer' 사용
    rag_result_text = selected_conv['answer']

    # 출력할 텍스트 라인 저장 리스트 및 처리한 관광지명 집합 초기화
    processed_output_lines = []
    processed_place_names = set()
    # 여행 계획표 텍스트를 별도로 저장할 변수 초기화
    table_plan_text = ""
    # 여행 계획 표 섹션 진입 여부를 나타내는 플래그
    in_plan_section = False

    # LLM 응답을 줄 단위로 순회하면서 처리
    for line in rag_result_text.split('\n'):
        # '상세 여행 계획' 섹션 시작을 감지 (단, 표 헤더 라인 제외)
        if "상세 여행 계획" in line and "일차 | 시간 | 활동" not in line:
            processed_output_lines.append(line)
            in_plan_section = True  # 이후부터는 여행 계획표 영역임을 표시
            continue

        if not in_plan_section:
            # 관광지 이름 패턴이 있으면 추출
            name_match = re.search(r"관광지 이름:\s*(.+)", line)
            if name_match:
                current_place_name = name_match.group(1).strip()
                # 아직 처리하지 않은 관광지면 처리 시작
                if current_place_name not in processed_place_names:
                    processed_output_lines.append(line)
                    processed_place_names.add(current_place_name)

                    # 관광지 이름에 맞는 위도/경도 데이터 조회
                    found_place_data = tour_data_df[
                        (tour_data_df['관광지명'].str.strip() == current_place_name) &
                        (pd.notna(tour_data_df['위도'])) &
                        (pd.notna(tour_data_df['경도']))
                    ]
                    
                    # 대화에서 저장된 사용자 위치 정보 가져오기
                    current_user_lat_conv = selected_conv.get('user_lat')
                    current_user_lon_conv = selected_conv.get('user_lon')

                    # 위치 데이터가 존재하면 거리 계산
                    if not found_place_data.empty and current_user_lat_conv is not None and current_user_lon_conv is not None:
                        place_lat = found_place_data['위도'].iloc[0]
                        place_lon = found_place_data['경도'].iloc[0]
                        distance = haversine(current_user_lat_conv, current_user_lon_conv, place_lat, place_lon)
                        # 거리 정보 출력 (소수점 2자리)
                        processed_output_lines.append(f"- 사용자 위치 기준 거리(km): 약 **{distance:.2f}** km")
                    else:
                        # 위치 데이터가 없거나 불일치할 경우 안내 메시지 출력
                        processed_output_lines.append("- 사용자 위치 기준 거리(km): 정보 없음 (데이터 불일치 또는 좌표 누락)")
                else:
                    # 이미 처리된 관광지 이름에 대한 중복 라인 중 거리 정보가 아니면 출력
                    if not re.search(r"거리\(km\):", line):
                        processed_output_lines.append(line)
            else:
                # 관광지 이름 패턴이 아니고, 거리 정보 라인도 아니면 일반 텍스트로 추가
                if not re.search(r"거리\(km\):", line): # 중복 거리 정보 방지
                    processed_output_lines.append(line)
        else:
            # 여행 계획표 영역에 들어오면 해당 라인을 여행 계획표 텍스트에 누적
            table_plan_text += line + "\n"

    # 처리된 일반 출력 텍스트 출력
    st.markdown("\n".join(processed_output_lines))

    # 여행 계획표가 있다면 DataFrame으로 파싱 후 시각화
    if table_plan_text.strip():
        try:
            plan_lines = table_plan_text.strip().split('\n')
            
            # 최소한 표의 기본 형식(헤더, 구분선, 데이터 행)이 맞는지 검증
            if len(plan_lines) >= 3 and plan_lines[0].count('|') >= 2 and plan_lines[1].count('|') >= 2 and all(re.match(r'^-+$', s.strip()) for s in plan_lines[1].split('|') if s.strip()):
                # 헤더 파싱 및 공백 제거
                header = [h.strip() for h in plan_lines[0].split('|') if h.strip()]
                data_rows = []
                for row_str in plan_lines[2:]:  # 헤더, 구분선 제외 데이터 행 순회
                    if row_str.strip() and row_str.startswith('|'):
                        # 각 컬럼 값 파싱 및 양쪽 빈 요소 제거
                        parsed_row = [d.strip() for d in row_str.split('|') if d.strip() or d == '']
                        # 첫 번째와 마지막 빈 문자열 제거 로직 개선 (공백으로만 된 스트링도 처리)
                        if parsed_row and not parsed_row[0].strip(): parsed_row = parsed_row[1:]
                        if parsed_row and not parsed_row[-1].strip(): parsed_row = parsed_row[:-1]

                        # 데이터 행의 길이가 헤더와 맞지 않으면 (일차 병합 등)
                        # 헤더 개수에 맞춰 데이터를 채워넣거나 자르는 보정 로직 추가
                        if len(parsed_row) < len(header):
                            parsed_row.extend([''] * (len(header) - len(parsed_row)))
                        elif len(parsed_row) > len(header):
                            parsed_row = parsed_row[:len(header)]
                        
                        data_rows.append(parsed_row)

                # 모든 데이터 행이 헤더 컬럼 수와 일치하는지 확인 (재확인)
                if data_rows and all(len(row) == len(header) for row in data_rows):
                    temp_plan_df = pd.DataFrame(data_rows, columns=header)
                    
                    # '일차' 컬럼이 있으면 중복된 일차 값을 빈 문자열로 바꿔서 시각적 병합 효과
                    if '일차' in temp_plan_df.columns:
                        last_day = ""
                        for i in range(len(temp_plan_df)):
                            current_day = temp_plan_df.loc[i, '일차']
                            if current_day == last_day and current_day != '':
                                temp_plan_df.loc[i, '일차'] = ''
                            else:
                                last_day = current_day
                    
                    st.markdown("---")
                    st.markdown("### 🗓️ AI가 제안하는 여행 계획표")
                    st.dataframe(temp_plan_df, use_container_width=True)
                else:
                    st.warning("AI가 생성한 여행 계획표 형식이 예상과 다릅니다. 원본 텍스트로 표시합니다.")
                    st.markdown(table_plan_text)
            else:
                st.warning("AI가 생성한 여행 계획표 형식이 예상과 다릅니다. 원본 텍스트로 표시합니다.")
                st.markdown(table_plan_text)
        except Exception as e:
            st.warning(f"여행 계획표를 파싱하는 중 오류가 발생했습니다: {e}. 원본 텍스트로 표시합니다.")
            st.markdown(table_plan_text)
