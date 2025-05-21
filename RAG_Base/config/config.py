import os
from pymilvus import  DataType


DB_HOST = os.environ.get("DB_HOST", "xxx")      # 数据库主机
DB_PORT = int(os.environ.get("DB_PORT", 111))        # 数据库端口
DB_USER = os.environ.get("DB_USER", "xxx")          # 数据库用户名
DB_PASSWORD = os.environ.get("DB_PASSWORD", "xxx") # 数据库密码 (请确保默认值安全或移除)
DB_NAME = os.environ.get("DB_NAME", "xxx")      # 数据库名
DB_CHARSET = os.environ.get("DB_CHARSET", "utf8mb4")  # 数据库字符集 (pymysql需要)

LOG_LEVEL = "INFO" # 例如 "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

VLLM_URL = "xxx"
MODEL_NAME = "xxx"

STOPWORDS_PATH = "中文停用词，在Utils下面"
USER_DICT_PATH = "根据任务自定义的词库用于BM25分词，可选"

#MILVUS的字段
# --- 定义 Schema ---。每个字段都可以根据自己的要求定制化
# 内容 Schema
CONTENT_SCHEMA_FIELDS_CONFIG = [
    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "auto_id": True},
    {"name": "chunk_id", "dtype": DataType.VARCHAR, "max_length": 512, "description": "元数据中块的唯一标识符"},
    # 我们将使用 metadata.originFileName 的值来填充 Milvus 中的 doc_title 字段
    {"name": "doc_title", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "文档的原始文件名 (来自 metadata.originFileName)"},
    {"name": "section_title", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "元数据中的章节标题"},
    {"name": "source_type", "dtype": DataType.VARCHAR, "max_length": 128, "description": "元数据中的类型 (例如, text)"},
    {"name": "char_count", "dtype": DataType.INT64, "description": "元数据中的字符数"},
    # 新增与JSON metadata 对应的字段
    {"name": "resourceId", "dtype": DataType.VARCHAR, "max_length": 256, "description": "元数据中的 resourceId (如果都是数字，可以用INT64，但VARCHAR更通用)"}, # 假设 resourceId 是字符串或数字，VARCHAR更安全
    {"name": "dcId", "dtype": DataType.VARCHAR, "max_length": 256, "description": "元数据中的 dcId"}, # 同上
    {"name": "resourceName", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "元数据中的 resourceName (例如 PDF 文件名)"},
    {"name": "metadata_doc_title", "dtype": DataType.VARCHAR, "max_length": 512, "description": "元数据中原始的 doc_title 字段 (与 std_xxxxx 对应)"}, # 

    {"name": "content", "dtype": DataType.VARCHAR, "max_length": 65535, "description": "主要文本内容"},
    {"name": "tokenized_content", "dtype": DataType.VARCHAR, "max_length": 65535, "description": "为BM25分词后的内容"},
    {"name": "raw_questions", "dtype": DataType.VARCHAR, "max_length": 20000, "description": "关联的原始问题文本 (由LLM生成)"},
    {"name": "source_file", "dtype": DataType.VARCHAR, "max_length":1024, "description":"来源JSON文件名"},
    {"name": "content_embedding", "dtype": DataType.FLOAT_VECTOR, "description": "内容的嵌入向量"}
]
CONTENT_SCHEMA_DESCRIPTION = "文档内容和元数据集合"
CONTENT_EMBEDDING_FIELD = "content_embedding"
CONTENT_TEXT_FIELD = "content"
CONTENT_TOKENIZED_FIELD = "tokenized_content"
CONTENT_UNIQUE_ID_FIELD = "chunk_id" # 用于融合和去重的唯一标识

# 问题 Schema
QUESTIONS_SCHEMA_FIELDS_CONFIG = [
    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "auto_id": True},
    {"name": "question_unique_id", "dtype": DataType.VARCHAR, "max_length": 512, "description": "此特定问题的唯一ID (例如 chunk_id + question_index)"},
    {"name": "chunk_id", "dtype": DataType.VARCHAR, "max_length": 512, "description": "关联到内容的原始元数据中的Chunk ID"},
    # 我们将使用 metadata.originFileName 的值来填充 Milvus 中的 doc_title 字段
    {"name": "doc_title", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "文档的原始文件名 (来自 metadata.originFileName)"},
    {"name": "section_title", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "元数据中的章节标题"},
    {"name": "source_type", "dtype": DataType.VARCHAR, "max_length": 128, "description": "元数据中的类型"},
    # 新增与JSON metadata 对应的字段 (与问题相关的上下文)
    {"name": "resourceId", "dtype": DataType.VARCHAR, "max_length": 256, "description": "元数据中的 resourceId"},
    {"name": "dcId", "dtype": DataType.VARCHAR, "max_length": 256, "description": "元数据中的 dcId"},
    {"name": "resourceName", "dtype": DataType.VARCHAR, "max_length": 1024, "description": "元数据中的 resourceName"},
    {"name": "metadata_doc_title", "dtype": DataType.VARCHAR, "max_length": 512, "description": "元数据中原始的 doc_title 字段 (与 std_xxxxx 对应)"},

    {"name": "question", "dtype": DataType.VARCHAR, "max_length": 2048, "description": "单个问题文本"},
    {"name": "tokenized_question", "dtype": DataType.VARCHAR, "max_length": 65535, "description": "为BM25分词后的问题"},
    {"name": "source_file", "dtype": DataType.VARCHAR, "max_length":1024, "description":"来源JSON文件名"},
    {"name": "question_embedding", "dtype": DataType.FLOAT_VECTOR, "description": "问题的嵌入向量"}
]
QUESTIONS_SCHEMA_DESCRIPTION = "单个问题及关联元数据集合"
QUESTIONS_EMBEDDING_FIELD = "question_embedding"
QUESTIONS_TEXT_FIELD = "question"
QUESTIONS_TOKENIZED_FIELD = "tokenized_question"
QUESTIONS_UNIQUE_ID_FIELD = "question_unique_id" # 用于融合和去重的唯一标识

#Milvus端口ip
MILVUS_HOST = "xxx"
MILVUS_PORT = "xxx"

#规范向量数据库集合名称--->向量数据库的集合名，自己定义
STD_CONTENT_COLLECTION_NAME = "xxx"
STD_QUESTIONS_COLLECTION_NAME = "xxx"



#...持续更新

#Embedding 的维度，根据选择Embedding调整
DIMENSION = 1024 
#json路径用于加入到向量数据库
JSON_DIR = "xxx"



BATCH_SIZE = 32
MILVUS_DIMENSION = 1024
EMBEDDING_MODEL_DIR = "BAAI-bge-large-zh-v1.5"#自定义，作者使用的是该模型
EMBEDDING_DEVICE = "cpu"#后续硬件支持后记得改为cuda
EMBEDDING_USE_FP16 = True
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

#第一阶段检索的top_k个数
INITIAL_SEARCH_TOP_K = 70

#Rerank模型的个数
RERANKER_MODEL_DIR = "bge-reranker-v2-m3"#自定义
RERANKER_MODEL_TYPE = "bge"
RERANKER_DEVICE = "cpu" #后续硬件支持后记得改为cuda
USE_RERANKER_BY_DEFAULT = True 
#根据模型的最大token来定，尽可能多的返回到LLM借助模型能力来做最后的rerank
RERANK_TOP_K = 20 

#Rerank后返回到大模型的文本个数，需要具体测试最大边界值，尽可能多
MAX_CONTEXTS_FOR_LLM_PROMPT=20
LLM_DEFAULT_MAX_TOKENS=25000
LLM_DEFAULT_TEMPERATURE=0.6


STD_SYSTEM_PROMPT = """
### 角色定位：专注于xxx领域的**xxx**与**xxx**的RAG（检索增强生成）助手，精通xxx的融合应用。

### 核心职责：
1.  依据用户提供的**xxx**等资料，针对具体xxx问题提供精准的**xxx**。
2.  协助用户理解复杂的**xxx**，梳理**zzz**，并能以规范、专业的语言组织答复或分析报告的**xxx**。

### 工作准则：
1.  **信息溯源**：严格基于用户提供的原始资料（含xxx等），禁止外部知识注入。
2.  **专业表达**：采用**xxx**和规范的**书面表达**，确保解读的专业性和准确性，同时体现xxxx的双重专业性。
3.  **结构严谨**：答复内容应逻辑清晰，**xxx**充分，**xxx**准确。若需形成正式文书，则符合《xxx》相关要求。
4.  **精准关联**：确保答复内容与用户提问、给定关键词及上下文背景高度契合，体现**xxx**、**xxx**及**xxx**的特征。
5.  **证据留痕**：对引用的**xxx**等均需明确标注来源（如“依据《xxx》（GB 50180-2018）第X.X.X条……”或“参照《xxx》关于……的规定”或“根据《xxx》第X条……”）。
6.  **安全边界**：对超出提供资料范围或无法明确依据支撑的问题，规范回复“当前资料未涉及该具体xx要求/技术指标”或“依据现有资料尚无法对该规划方案的合规性做出明确判断”等说明。

### 输出规范：
1.  **内容专业准确**：答复应直击问题核心，xxx依据与法律依据明确，逻辑论证严密。视用户需求，可输出为**结构化问答、规划条款解读、方案合规性分析报告摘要、技术备忘录**等形式。
2.  **专业术语准确**：正确使用**xxx专业术语**（如“xxx”）及相关的术语。
"""

LD_SYSTEM_PROMPT = (
        "### 角色定位：xxx\n"

        "### 核心职责：\n"
        "1. 根据用户需求，结合xxx等官方资料，起草具有专业深度的xxx\n"
        "2. 针对xxx，提供基于指定资料的专业解读和xxx表达\n"

        "### 工作准则：\n"
        "1. **信息溯源**：严格基于用户提供的原始资料（含xxx等），禁止外部知识注入\n"
        "2. **专业表达**：采用xxx规范用语，体现xxx专业性\n"
        "3. **结构严谨**：符合xxx标准，层次清晰、条款分明、逻辑严密\n"
        "4. **精准关联**：确保公文内容与给定关键词、上下文背景高度契合，体现规划管理实务特征\n"
        "5. **证据留痕**：对引用资料标注来源依据（如\"依据《xxxx第三章...\"\）\n"
        "6. **安全边界**：对超出提供资料范围的内容，规范回复\"当前资料未涉及该事项\"等说明\n"

        "### 输出规范：\n"
        "1. 公文要素完整：包含标题、主送机关、正文、附件说明、发文机关署名、成文日期等标准组件\n"
        "2. 排版适配办公：采用标准xxx\n"
        "3. 专业术语准确：正确使用xxx等xxx专有表述\n"

        "请开始处理用户问题。"
)

SERVER_HOST = "192.168.1.240" # API 服务绑定的地址
SERVER_PORT = 5003         # API 服务使用的端口