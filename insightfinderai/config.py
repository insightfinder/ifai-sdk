DEFAULT_API_URL = "https://ai.insightfinder.com"

# API Endpoints
# Chatbot endpoints
CHATBOT_ENDPOINT = "api/external/v1/chatbot/stream-with-type"
SET_SYSTEM_PROMPT_ENDPOINT = "api/external/v1/chatbot/sysprompt/get-sysprompt"
APPLY_SYSTEM_PROMPT_ENDPOINT = "api/external/v1/chatbot/sysprompt/apply-sysprompt"
CLEAR_SYSTEM_PROMPT_ENDPOINT = "api/external/v1/chatbot/sysprompt/clear-sysprompt"
NEW_CHAT_SESSION_ENDPOINT = "api/external/v1/chatbot/new-chat-session"
TRACE_PROJECT_NAME_ENDPOINT = "api/external/v1/chatbot/get-trace-project-name-v2"
MODEL_INFO_ENDPOINT = "api/external/v1/chatbot/model-info"
MODEL_INFO_LIST_ENDPOINT = "api/external/v1/chatbot/model-info-list"

# Session management endpoints
CREATE_SESSION_ENDPOINT = "api/external/v1/chatbot/new-chatbot"
DELETE_SESSION_ENDPOINT = "api/external/v1/chatbot"
SUPPORTED_MODELS_ENDPOINT = "api/external/v1/chatbot/supported-list"

# Evaluation endpoints
EVALUATION_ENDPOINT = "api/external/v1/evaluation/bias-hallu"
SAFETY_EVALUATION_ENDPOINT = "api/external/v1/evaluation/safety"

# Other endpoints
ORG_TOKEN_USAGE_ENDPOINT = "api/external/v1/llm-labs/current-token-map"

# Real model endpoints
REAL_MODEL_LIST_SEARCH_ENDPOINT = "api/external/v1/real-model/model-list-search"
REAL_MODEL_LIST_SEARCH_WITH_DATASET_ENDPOINT = "api/external/v1/real-model/model-list-search-withdataset"

# Dataset endpoints
DATASET_LIST_ENDPOINT = "api/external/v1/llm-lab/datasets"
DATASET_SEARCH_ENDPOINT = "api/external/v1/llm-lab/datasets/search"

# Prompt template endpoints
PROMPT_TEMPLATE_VERSIONS_ENDPOINT = "api/external/v2/prompt-templates/versions"
PROMPT_TEMPLATE_LATEST_PROMPTS_ENDPOINT = "api/external/v2/prompt-templates/latest-version-prompts"
PROMPT_TEMPLATE_FROM_LIST_ENDPOINT = "api/external/v2/prompt-templates/from-list"
PROMPT_TEMPLATE_BY_VERSION_ENDPOINT = "api/external/v2/prompt-templates/template-version-id"

# Template compare endpoints
TEMPLATE_COMPARE_RUN_ENDPOINT = "api/external/v1/llm-lab/template-compare/compare"
TEMPLATE_COMPARE_DETAIL_ENDPOINT = "api/external/v1/llm-lab/template-compare/detail"
TEMPLATE_COMPARE_EVALUATION_ENDPOINT = "api/external/v1/llm-lab/template-compare/evaluation-detail"
TEMPLATE_COMPARE_WINNER_ENDPOINT = "api/external/v1/llm-lab/template-compare/winner"

# Customer infrastructure endpoints
CUSTOMER_INFRA_OPTIONS_ENDPOINT = "api/external/v1/customer-infra/options"
CUSTOMER_INFRA_SETTINGS_ENDPOINT = "api/external/v1/customer-infra/settings"
CUSTOMER_INFRA_SWITCH_ENDPOINT = "api/external/v1/customer-infra/switch"
CUSTOMER_INFRA_VERIFY_TOKEN_ENDPOINT = "api/external/v1/customer-infra/verify-token"
CUSTOMER_INFRA_COMPARE_ENDPOINT = "api/external/v1/customer-infra/compare"
CUSTOMER_INFRA_INGEST_PLAYBOOK_ENDPOINT = "api/external/v1/customer-infra/ingest-playbook"
CUSTOMER_INFRA_INGEST_MATTER_ENDPOINT = "api/external/v1/customer-infra/ingest-matter"
CUSTOMER_INFRA_MATTERS_BASE_ENDPOINT = "api/external/v1/customer-infra/matters"
CUSTOMER_INFRA_DATASETS_ENDPOINT = "api/external/v1/customer-infra/customer-datasets"
CUSTOMER_INFRA_DATASET_UPLOAD_FIELDS_ENDPOINT = "api/external/v1/customer-infra/dataset-upload-fields"
CUSTOMER_INFRA_PROMPT_LIBRARY_UPLOAD_FIELDS_ENDPOINT = "api/external/v1/customer-infra/prompt-library-upload-fields"