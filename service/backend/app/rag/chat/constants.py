MAX_CHAT_DOCUMENTS = 6
MAX_CHAT_HISTORY = 8
MAX_CHAT_EXCERPT_CHARS = 900
LOCAL_CHAT_MODEL = "local_rag_chat_guard"
LOCAL_DIAGNOSIS_HISTORY_MODEL = "local_diagnosis_history_reporter"
LOCAL_RAG_EVIDENCE_FALLBACK_MODEL = "local_rag_evidence_fallback"
ANSWER_MODE_RAG_EVIDENCE = "rag_evidence"
ANSWER_MODE_GENERAL_DOMAIN = "general_domain"
ANSWER_MODE_OUT_OF_SCOPE = "out_of_scope"
ANSWER_MODE_DIAGNOSIS_HISTORY = "diagnosis_history"
NO_RAG_EVIDENCE_NOTICE = "검색 근거 없음"
NON_DOMAIN_CHAT_ANSWER = (
    "이 챗은 부분방전 진단 RAG 전용입니다.\n\n"
    "- 코로나 방전, 보이드 방전, 표면 방전, 정상 판단처럼 진단 데이터와 연결된 질문을 입력하면 근거 문서를 검색합니다.\n"
    "- 일반 지식 질문은 RAG 근거를 붙이지 않습니다."
)

GREETING_QUERIES = {
    "ㅎㅇ",
    "하이",
    "안녕",
    "안녕하세요",
    "hello",
    "hi",
    "hey",
}

DOMAIN_TERMS = (
    "부분방전",
    "방전",
    "정상",
    "코로나",
    "보이드",
    "표면",
    "노이즈",
    "hfct",
    "prpd",
    "prps",
    "절연",
    "전압",
    "전류",
    "설비",
    "변압기",
    "acsr",
    "gis",
    "센서",
    "결함",
    "피크",
    "진단",
    "판단",
    "근거",
    "위상",
    "패턴",
    "이상",
    "고장",
    "열화",
    "데이터셋",
    "rag",
)
