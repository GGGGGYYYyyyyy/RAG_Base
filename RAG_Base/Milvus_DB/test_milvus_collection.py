# test_collection_connections.py
import logging
import sys
import os
from typing import Optional,List,Dict

# --- 依赖项导入 ---
try:
    # 确保路径正确，指向包含 MilvusManager 和 Schema 定义的文件
    from .Milvus_Config import ( 
        MilvusManager,
        MilvusException,
        DataType, # Import DataType if used in schema definition check below
        # 内容集合相关常量
        CONTENT_SCHEMA_FIELDS_CONFIG,
        CONTENT_SCHEMA_DESCRIPTION,
        CONTENT_EMBEDDING_FIELD,
        CONTENT_TEXT_FIELD,
        CONTENT_TOKENIZED_FIELD,
        CONTENT_UNIQUE_ID_FIELD,
        # 问题集合相关常量
        QUESTIONS_SCHEMA_FIELDS_CONFIG,
        QUESTIONS_SCHEMA_DESCRIPTION,
        QUESTIONS_EMBEDDING_FIELD,
        QUESTIONS_TEXT_FIELD,
        QUESTIONS_TOKENIZED_FIELD,
        QUESTIONS_UNIQUE_ID_FIELD
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 Milvus_DB/Milvus_Config.py (或您的模块) 在 Python 路径中，并包含所有必要的定义。")
    # 尝试导入 MilvusException 以便后续使用
    try:
        from pymilvus import MilvusException, DataType
    except ImportError:
        print("警告: 无法从 pymilvus 导入 MilvusException 或 DataType。")
        class MilvusException(Exception): pass
        class DataType: # Dummy DataType
            INT64 = 5
            VARCHAR = 21
            FLOAT_VECTOR = 101
    # 定义默认 Schema 以允许脚本尝试运行，但结果可能不准确
    CONTENT_SCHEMA_FIELDS_CONFIG = []
    QUESTIONS_SCHEMA_FIELDS_CONFIG = []
    CONTENT_UNIQUE_ID_FIELD = "chunk_id"
    CONTENT_EMBEDDING_FIELD = "content_embedding"
    CONTENT_TOKENIZED_FIELD = "tokenized_content"
    CONTENT_TEXT_FIELD = "content"
    QUESTIONS_UNIQUE_ID_FIELD = "question_unique_id"
    QUESTIONS_EMBEDDING_FIELD = "question_embedding"
    QUESTIONS_TOKENIZED_FIELD = "tokenized_question"
    QUESTIONS_TEXT_FIELD = "question"
    CONTENT_SCHEMA_DESCRIPTION = "内容（默认）"
    QUESTIONS_SCHEMA_DESCRIPTION = "问题（默认）"
    # exit(1) # 可以选择在这里退出

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
test_logger = logging.getLogger("集合连接测试脚本")

# --- 配置参数 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
CONTENT_COLLECTION_NAME = "LeaderSpeak_contents_v3" # 确认名称正确
QUESTIONS_COLLECTION_NAME = "LeaderSpeak_questions_v3" # 确认名称正确
DIMENSION = 1024 # 确认维度正确

# 辅助文件路径（在仅测试连接/加载时非必需，但 MilvusManager 可能需要）
STOPWORDS_PATH = "/home/cdipd-admin/RAG_LeaderSpeak/Utils/cn_stopWord.txt" # 使用实际路径或 None
USER_DICT_PATH = "/home/cdipd-admin/RAG_LeaderSpeak/Utils/规划词库V6.txt" # 使用实际路径或 None

def test_connection(collection_name: str, 
                    schema_config: List[Dict], 
                    schema_desc: str, 
                    embedding_field: str,
                    text_field: str,
                    tokenized_field: str,
                    unique_id_field: str):
    """尝试初始化并加载指定集合的 MilvusManager。"""
    manager: Optional[MilvusManager] = None
    test_logger.info(f"--- 正在测试集合: {collection_name} ---")
    try:
        # 1. 初始化管理器
        test_logger.info(f"尝试初始化管理器...")
        manager = MilvusManager(
            dimension=DIMENSION,
            collection_name=collection_name,
            schema_fields_config=schema_config,
            schema_description=schema_desc,
            embedding_field_name=embedding_field,
            text_field_name_for_bm25=text_field,
            tokenized_text_field_name_for_bm25=tokenized_field,
            unique_id_field_for_fusion=unique_id_field,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            stopwords_path=STOPWORDS_PATH,
            user_dict_path=USER_DICT_PATH,
            log_level=logging.INFO # 可以设为 DEBUG 看 MilvusManager 内部日志
        )
        test_logger.info(f"管理器初始化调用完成 (不代表成功加载现有集合)。")

        # 2. 显式尝试加载集合到内存
        # 这一步会触发对已存在集合的 Schema 检查（如果初始化时未完成）
        # 并实际将数据加载到 QueryNode
        test_logger.info(f"尝试加载集合到内存...")
        if manager.collection is None:
             # _check_and_load_collection 在 __init__ 中已经失败了
             # 并且 load_collection 内部会检查 self.collection is None
             # 所以这里调用 load_collection() 会直接触发那个 ValueError
             manager.load_collection() # 这会引发 ValueError
        elif not manager.is_loaded():
             manager.load_collection() # 尝试加载
        
        # 3. 检查加载状态
        if manager.is_loaded():
             test_logger.info(f"成功: 集合 '{collection_name}' 已连接并加载到内存。")
             # 可以选择性地获取并打印统计信息
             stats = manager.get_collection_stats()
             if stats:
                 test_logger.info(f"  集合统计: {stats['num_entities']} 个实体。")
             return True
        else:
             # 如果 manager.collection 存在但 is_loaded() 返回 False，说明 load_collection 可能静默失败
             test_logger.error(f"失败: 集合 '{collection_name}' 存在但未能成功加载到内存（is_loaded 返回 False）。")
             return False

    except ValueError as ve: # 通常是因为 self.collection is None
        test_logger.error(f"失败: 初始化或加载集合 '{collection_name}' 时捕获到 ValueError: {ve}")
        return False
    except MilvusException as me: # 捕获 Milvus 特定异常，包括 SchemaNotReadyException
        test_logger.error(f"失败: 初始化或加载集合 '{collection_name}' 时捕获到 MilvusException: {me.message} (code={me.code})")
        # if me.code == 1: # SchemaNotReadyException 的 code 通常是 1
        #     test_logger.error("===> 这很可能是 Schema 不匹配的问题！请仔细核对代码中的 Schema 定义与 Milvus 中实际存储的是否完全一致（特别是移除 default_value 后）。")
        return False
    except Exception as e:
        test_logger.error(f"失败: 初始化或加载集合 '{collection_name}' 时发生意外错误: {e}", exc_info=True)
        return False
    finally:
        if manager:
            manager.disconnect()
            test_logger.info(f"集合 '{collection_name}' 的连接已断开。")

if __name__ == "__main__":
    test_logger.info("开始执行 Milvus 双集合连接测试...")

    # 测试内容集合
    content_success = test_connection(
        collection_name=CONTENT_COLLECTION_NAME,
        schema_config=CONTENT_SCHEMA_FIELDS_CONFIG,
        schema_desc=CONTENT_SCHEMA_DESCRIPTION,
        embedding_field=CONTENT_EMBEDDING_FIELD,
        text_field=CONTENT_TEXT_FIELD,
        tokenized_field=CONTENT_TOKENIZED_FIELD,
        unique_id_field=CONTENT_UNIQUE_ID_FIELD
    )

    print("-" * 60) # 分隔符

    # 测试问题集合
    questions_success = test_connection(
        collection_name=QUESTIONS_COLLECTION_NAME,
        schema_config=QUESTIONS_SCHEMA_FIELDS_CONFIG,
        schema_desc=QUESTIONS_SCHEMA_DESCRIPTION,
        embedding_field=QUESTIONS_EMBEDDING_FIELD,
        text_field=QUESTIONS_TEXT_FIELD,
        tokenized_field=QUESTIONS_TOKENIZED_FIELD,
        unique_id_field=QUESTIONS_UNIQUE_ID_FIELD
    )

    print("-" * 60)
    test_logger.info("--- 测试总结 ---")
    test_logger.info(f"内容集合 ({CONTENT_COLLECTION_NAME}) 连接和加载: {'成功' if content_success else '失败'}")
    test_logger.info(f"问题集合 ({QUESTIONS_COLLECTION_NAME}) 连接和加载: {'成功' if questions_success else '失败'}")

    if not content_success or not questions_success:
        test_logger.error("存在集合连接或加载失败。请检查上面的错误日志。最常见的原因是 Schema 不匹配。")
        test_logger.error("强烈建议：")
        test_logger.error("1. 仔细核对 Milvus_DB/Milvus_Config.py 中的 Schema 定义，确保完全移除了 default_value 且与预期一致。")
        test_logger.error("2. 使用数据加载脚本 (如 Milvus_Create.py) 并设置 drop_existing=True 来强制重建集合。")
    else:
        test_logger.info("所有集合连接和加载测试通过！")