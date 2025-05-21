# -*- coding: utf-8 -*-
import os
import sys
import glob
import logging # 导入日志库
import sys
import os
import sys
import os
print("--- Query/test_query.py sys.path ---")
for p in sys.path:
    print(p)
try:
    import Milvus_DB.Milvus_Config 
    print(f"!!! Query/test_query.py is loading Milvus_Config from: {Milvus_DB.Milvus_Config.__file__} !!!")
except ImportError:
     print("Query/test_query.py could not import Milvus_DB.Milvus_Config directly.")
     try:
         from Milvus_DB import Milvus_Config # 尝试相对导入
         print(f"!!! Query/test_query.py is loading Milvus_Config via relative path from: {Milvus_Config.__file__} !!!")
     except ImportError:
          print("Query/test_query.py could not import Milvus_Config using relative path either.")
     except Exception as e_attr: # 捕获可能的其他错误
         print(f"Error trying relative import: {e_attr}")
except AttributeError:
     print("Milvus_DB.Milvus_Config might be loaded differently (e.g., namespace package).")
except Exception as e_load: # 捕获加载时的其他错误
     print(f"Error importing/accessing Milvus_DB.Milvus_Config: {e_load}")

print("-" * 30)
try:
    from Embedding.embedding_generator import EmbeddingGenerator
    from config.config import (
        CONTENT_SCHEMA_FIELDS_CONFIG, 
        CONTENT_SCHEMA_DESCRIPTION, 
        CONTENT_EMBEDDING_FIELD, 
        CONTENT_TEXT_FIELD, 
        CONTENT_TOKENIZED_FIELD, 
        CONTENT_UNIQUE_ID_FIELD,
                        
        QUESTIONS_SCHEMA_FIELDS_CONFIG,
        QUESTIONS_SCHEMA_DESCRIPTION,
        QUESTIONS_EMBEDDING_FIELD,
        QUESTIONS_TEXT_FIELD,
        QUESTIONS_TOKENIZED_FIELD, 
        QUESTIONS_UNIQUE_ID_FIELD,

        MILVUS_HOST,
        MILVUS_PORT,
        STD_CONTENT_COLLECTION_NAME,
        STD_QUESTIONS_COLLECTION_NAME,
        DIMENSION,
        BATCH_SIZE,
        STOPWORDS_PATH,
        USER_DICT_PATH,
        JSON_DIR,

        EMBEDDING_MODEL_DIR,
        EMBEDDING_DEVICE
    )
    from .Milvus_Config import (
        MilvusManager
        # load_and_prepare_data_from_json
    )
except ImportError as e:
    try:
        from pymilvus import MilvusException
    except ImportError:
        pass 
    print(f"导入错误: {e}。请确保所有必要的模块 (包括 milvus_toolkit.py 和 EmbeddingGenerator) 都在 Python 路径中。")
   
from Milvus_DB.milvus_data_helpers import load_and_prepare_data_from_json
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO, # 设置默认日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
main_logger = logging.getLogger("主脚本")

# --- 初始化嵌入模型 ---
main_logger.info(f"--- 正在初始化嵌入模型 ({EMBEDDING_MODEL_DIR}) ---")
embedding_generator = None # 先声明
try:
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_DIR, EMBEDDING_DEVICE, use_fp16=True if EMBEDDING_MODEL_DIR == "cuda" else False)
    if not hasattr(embedding_generator, 'encode_documents'):
         main_logger.warning("EmbeddingGenerator 可能没有 'encode_documents' 方法，load_and_prepare_data_from_json 可能需要适配。")
    main_logger.info(f"嵌入模型加载成功，使用设备: {EMBEDDING_MODEL_DIR}")
except Exception as e:
    main_logger.critical(f"加载嵌入模型失败: {e}", exc_info=True)
    sys.exit(1)

# --- 初始化 Milvus 管理器 ---
main_logger.info(f"--- 正在初始化 Milvus 管理器 ---")
content_manager = None
questions_manager = None
try:
    main_logger.info(f"为内容集合 '{STD_CONTENT_COLLECTION_NAME}' 初始化管理器...")
    content_manager = MilvusManager(
        dimension=DIMENSION,
        collection_name=STD_CONTENT_COLLECTION_NAME,
        schema_fields_config=CONTENT_SCHEMA_FIELDS_CONFIG,
        schema_description=CONTENT_SCHEMA_DESCRIPTION,
        embedding_field_name=CONTENT_EMBEDDING_FIELD,
        text_field_name_for_bm25=CONTENT_TEXT_FIELD,
        tokenized_text_field_name_for_bm25=CONTENT_TOKENIZED_FIELD,
        unique_id_field_for_fusion=CONTENT_UNIQUE_ID_FIELD,
        host=MILVUS_HOST, 
        port=MILVUS_PORT,
        stopwords_path=STOPWORDS_PATH, 
        user_dict_path=USER_DICT_PATH,
        log_level=logging.INFO
    )
    main_logger.info(f"内容集合管理器 '{STD_CONTENT_COLLECTION_NAME}' 初始化成功。")

    main_logger.info(f"为问题集合 '{STD_QUESTIONS_COLLECTION_NAME}' 初始化管理器...")
    questions_manager = MilvusManager(
        dimension=DIMENSION,
        collection_name=STD_QUESTIONS_COLLECTION_NAME,
        schema_fields_config=QUESTIONS_SCHEMA_FIELDS_CONFIG,
        schema_description=QUESTIONS_SCHEMA_DESCRIPTION,
        embedding_field_name=QUESTIONS_EMBEDDING_FIELD,
        text_field_name_for_bm25=QUESTIONS_TEXT_FIELD,
        tokenized_text_field_name_for_bm25=QUESTIONS_TOKENIZED_FIELD,
        unique_id_field_for_fusion=QUESTIONS_UNIQUE_ID_FIELD,
        host=MILVUS_HOST, 
        port=MILVUS_PORT,
        stopwords_path=STOPWORDS_PATH, 
        user_dict_path=USER_DICT_PATH,
        log_level=logging.INFO
    )
    main_logger.info(f"问题集合管理器 '{STD_QUESTIONS_COLLECTION_NAME}' 初始化成功。")

except Exception as e:
    main_logger.critical(f"初始化 Milvus 管理器失败: {e}", exc_info=True)
    if content_manager: content_manager.disconnect()
    if questions_manager: questions_manager.disconnect()
    sys.exit(1)

# --- 创建/检查集合和索引，并加载到内存 ---
main_logger.info("--- 准备 Milvus 集合、索引并加载到内存 ---")
try:
    drop_existing = True
    
    main_logger.info(f"为内容集合 '{STD_CONTENT_COLLECTION_NAME}' 创建集合、索引并加载...")
    content_manager.create_collection(drop_existing=drop_existing, scalar_index_fields=["doc_title", "source_file"])
    content_manager.create_vector_index(index_type="HNSW", metric_type="COSINE")
    content_manager.load_collection()
    main_logger.info(f"内容集合 '{STD_CONTENT_COLLECTION_NAME}' 已成功加载到内存。")

    main_logger.info(f"为问题集合 '{STD_QUESTIONS_COLLECTION_NAME}' 创建集合、索引并加载...")
    questions_manager.create_collection(drop_existing=drop_existing, scalar_index_fields=["chunk_id", "source_file"])
    questions_manager.create_vector_index(index_type="HNSW", metric_type="COSINE")
    questions_manager.load_collection()
    main_logger.info(f"问题集合 '{STD_QUESTIONS_COLLECTION_NAME}' 已成功加载到内存。")

except Exception as e:
    main_logger.critical(f"创建集合、索引或加载集合到内存时出错: {e}", exc_info=True)
    if content_manager: content_manager.disconnect()
    if questions_manager: questions_manager.disconnect()
    sys.exit(1)

# --- 处理目录中的所有 JSON 文件 ---
main_logger.info(f"--- 开始处理目录 '{JSON_DIR}' 中的 JSON 文件 ---")

total_files_processed = 0
succeeded_files = 0
failed_files_list = []
total_content_items_added = 0
total_question_items_added = 0
total_files_to_process = 0 # 初始化

try:
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    total_files_to_process = len(json_files)
    main_logger.info(f"在目录 '{JSON_DIR}' 中找到 {total_files_to_process} 个 JSON 文件。")

    if total_files_to_process == 0:
        main_logger.warning("未找到任何 JSON 文件，程序将退出。")
    else:
        for i, json_path in enumerate(json_files):
            total_files_processed += 1
            file_basename = os.path.basename(json_path)
            main_logger.info(f"--- 正在处理文件 {i+1}/{total_files_to_process}: {file_basename} ---")
            
            file_failed = False
            try:
                # 1. 处理内容
                main_logger.info(f"为文件 '{file_basename}' 加载并准备 'content' 数据...")
                prepared_content_data, content_embeddings = load_and_prepare_data_from_json(
                    json_path=json_path,
                    embedding_model=embedding_generator,
                    target_schema_type="content",
                    batch_size=BATCH_SIZE
                )
                if prepared_content_data and content_embeddings is not None and len(content_embeddings) > 0 :
                    main_logger.info(f"准备向内容集合添加 {len(prepared_content_data)} 条数据 (来自文件 {file_basename})...")
                    existing_ids = set()
                    # 仅当集合存在且已有实体时才进行去重查询
                    if content_manager.collection and content_manager.collection.num_entities > 0:
                        ids_to_check = [item[CONTENT_UNIQUE_ID_FIELD] for item in prepared_content_data if CONTENT_UNIQUE_ID_FIELD in item]
                        if ids_to_check:
                            batch_size_dedup_in_op = 512 # 调整批次大小
                            for chunk_i in range(0, len(ids_to_check), batch_size_dedup_in_op):
                                batch_ids_for_in = ids_to_check[chunk_i : chunk_i + batch_size_dedup_in_op]
                                quoted_batch_ids = [f"'{str(id_val).replace("'", "''")}'" for id_val in batch_ids_for_in]
                                query_expr_dedup = f"{CONTENT_UNIQUE_ID_FIELD} in [{', '.join(quoted_batch_ids)}]"
                                log_expr_dedup_short = query_expr_dedup[:200] + "..." if len(query_expr_dedup) > 200 else query_expr_dedup
                                main_logger.debug(f"内容去重查询 (批次 {chunk_i//batch_size_dedup_in_op + 1}) 表达式 (片段): {log_expr_dedup_short}")
                                try:
                                    existing_docs = content_manager.query(
                                        expr=query_expr_dedup, 
                                        output_fields=[CONTENT_UNIQUE_ID_FIELD], 
                                        limit=len(batch_ids_for_in)
                                    )
                                    for doc in existing_docs: existing_ids.add(doc[CONTENT_UNIQUE_ID_FIELD])
                                except NameError: # MilvusException 可能未成功导入
                                     main_logger.warning(f"内容去重查询时捕获到 NameError (MilvusException 可能未导入)，继续执行但可能重复。")
                                except Exception as query_e: # 通用异常捕获
                                    main_logger.warning(f"内容去重查询失败 (批次 {chunk_i//batch_size_dedup_in_op + 1}): {type(query_e).__name__} - {str(query_e)[:100]}", exc_info=False)
                    
                    content_to_add_final = []
                    embeddings_for_add_final_c = []
                    for idx, data_item in enumerate(prepared_content_data):
                        if data_item[CONTENT_UNIQUE_ID_FIELD] not in existing_ids:
                            content_to_add_final.append(data_item)
                            embeddings_for_add_final_c.append(content_embeddings[idx])
                        else:
                            main_logger.debug(f"跳过已存在的内容块: {data_item[CONTENT_UNIQUE_ID_FIELD]}")
                    
                    if content_to_add_final:
                        insert_result_content = content_manager.add_data(embeddings_for_add_final_c, content_to_add_final)
                        if insert_result_content and insert_result_content.insert_count > 0:
                            total_content_items_added += insert_result_content.insert_count
                            main_logger.info(f"成功向内容集合添加 {insert_result_content.insert_count} 条数据 (来自 {file_basename})。")
                        elif insert_result_content and insert_result_content.insert_count == 0 :
                             main_logger.warning(f"向内容集合添加数据时未插入任何条目 (来自 {file_basename})，尽管尝试了添加。")
                    else:
                        main_logger.info(f"内容数据去重后或过滤后无新条目可添加 (来自 {file_basename})。")
                elif content_embeddings is None and not prepared_content_data:
                    main_logger.error(f"为文件 '{file_basename}' 加载或嵌入 'content' 数据失败，已跳过内容部分。")
                
                # 2. 处理问题
                main_logger.info(f"为文件 '{file_basename}' 加载并准备 'questions' 数据...")
                prepared_question_data, question_embeddings = load_and_prepare_data_from_json(
                    json_path=json_path,
                    embedding_model=embedding_generator,
                    target_schema_type="questions",
                    batch_size=BATCH_SIZE
                )
                if prepared_question_data and question_embeddings is not None and len(question_embeddings) > 0:
                    main_logger.info(f"准备向问题集合添加 {len(prepared_question_data)} 条数据 (来自文件 {file_basename})...")
                    existing_q_ids = set()
                    if questions_manager.collection and questions_manager.collection.num_entities > 0:
                        q_ids_to_check = [item[QUESTIONS_UNIQUE_ID_FIELD] for item in prepared_question_data if QUESTIONS_UNIQUE_ID_FIELD in item]
                        if q_ids_to_check:
                            batch_size_dedup_in_op_q = 512
                            for chunk_i in range(0, len(q_ids_to_check), batch_size_dedup_in_op_q):
                                batch_ids_for_in_q = q_ids_to_check[chunk_i : chunk_i + batch_size_dedup_in_op_q]
                                quoted_batch_ids_q = [f"'{str(id_val).replace("'", "''")}'" for id_val in batch_ids_for_in_q]
                                query_expr_dedup_q = f"{QUESTIONS_UNIQUE_ID_FIELD} in [{', '.join(quoted_batch_ids_q)}]"
                                log_expr_dedup_q_short = query_expr_dedup_q[:200] + "..." if len(query_expr_dedup_q) > 200 else query_expr_dedup_q
                                main_logger.debug(f"问题去重查询 (批次 {chunk_i//batch_size_dedup_in_op_q + 1}) 表达式 (片段): {log_expr_dedup_q_short}")
                                try:
                                    existing_q_docs = questions_manager.query(
                                        expr=query_expr_dedup_q, 
                                        output_fields=[QUESTIONS_UNIQUE_ID_FIELD], 
                                        limit=len(batch_ids_for_in_q)
                                    )
                                    for doc in existing_q_docs: existing_q_ids.add(doc[QUESTIONS_UNIQUE_ID_FIELD])
                                except NameError:
                                    main_logger.warning(f"问题去重查询时捕获到 NameError (MilvusException 可能未导入)，继续执行但可能重复。")
                                except Exception as query_e_q:
                                     main_logger.warning(f"问题去重查询失败 (批次 {chunk_i//batch_size_dedup_in_op_q + 1}): {type(query_e_q).__name__} - {str(query_e_q)[:100]}", exc_info=False)
                    
                    questions_to_add_final = []
                    embeddings_for_add_final_q = []
                    for idx, q_data_item in enumerate(prepared_question_data):
                        if q_data_item[QUESTIONS_UNIQUE_ID_FIELD] not in existing_q_ids:
                            questions_to_add_final.append(q_data_item)
                            embeddings_for_add_final_q.append(question_embeddings[idx])
                        else:
                            main_logger.debug(f"跳过已存在的问题: {q_data_item[QUESTIONS_UNIQUE_ID_FIELD]}")
                    if questions_to_add_final:
                        insert_result_questions = questions_manager.add_data(embeddings_for_add_final_q, questions_to_add_final)
                        if insert_result_questions and insert_result_questions.insert_count > 0:
                            total_question_items_added += insert_result_questions.insert_count
                            main_logger.info(f"成功向问题集合添加 {insert_result_questions.insert_count} 条数据 (来自 {file_basename})。")
                        elif insert_result_questions and insert_result_questions.insert_count == 0:
                            main_logger.warning(f"向问题集合添加数据时未插入任何条目 (来自 {file_basename})，尽管尝试了添加。")
                    else:
                        main_logger.info(f"问题数据去重后或过滤后无新条目可添加 (来自 {file_basename})。")
                elif question_embeddings is None and not prepared_question_data:
                    if prepared_question_data:
                         main_logger.error(f"文件 '{file_basename}' 的 'questions' 数据准备异常。")
                         file_failed = True
                    else:
                         main_logger.info(f"文件 '{file_basename}' 未找到有效 'questions' 数据或加载失败，跳过问题部分。")

                if not file_failed:
                    succeeded_files += 1
                    main_logger.info(f"文件 {file_basename} 处理完成。")

            except Exception as e_file_processing:
                main_logger.error(f"处理文件 {file_basename} 过程中发生严重错误: {e_file_processing}", exc_info=True)
                if not file_failed: file_failed = True
                is_already_in_failed = any(f_name == file_basename for f_name, _ in failed_files_list)
                if not is_already_in_failed:
                    failed_files_list.append((file_basename, str(e_file_processing)))
            
            if file_failed: # 确保文件被正确标记为失败
                is_already_in_failed = any(f_name == file_basename for f_name, _ in failed_files_list)
                if not is_already_in_failed:
                     failed_files_list.append((file_basename, "一个或多个处理步骤失败（详情请查看上方日志）"))

except KeyboardInterrupt:
     main_logger.warning("收到中断信号，正在停止处理...")
except Exception as e_outer_loop:
     main_logger.critical(f"处理目录时发生严重错误: {e_outer_loop}", exc_info=True)
finally:
    main_logger.info("--- 数据处理完成 ---")
    final_failed_files_count = len(failed_files_list)
    stats = {
        'files_processed_attempted': total_files_to_process,
        'files_actually_iterated': total_files_processed,
        'files_succeeded': succeeded_files,
        'files_failed': final_failed_files_count,
        'total_content_items_added': total_content_items_added,
        'total_question_items_added': total_question_items_added,
        'failed_files_details': failed_files_list
    }

    main_logger.info("\n处理结果统计:")
    main_logger.info(f"  尝试处理文件总数: {stats['files_processed_attempted']}")
    main_logger.info(f"  实际迭代处理文件数: {stats['files_actually_iterated']}")
    main_logger.info(f"  成功处理文件数: {stats['files_succeeded']}")
    main_logger.info(f"  处理失败文件数: {stats['files_failed']}")
    main_logger.info(f"  添加到 Milvus 的内容条目总数: {stats['total_content_items_added']}")
    main_logger.info(f"  添加到 Milvus 的问题条目总数: {stats['total_question_items_added']}")

    if stats['files_failed'] > 0:
        main_logger.warning("失败文件列表详情:")
        for file, error_msg in stats['failed_files_details']:
            main_logger.warning(f"  - 文件: {file}, 原因: {error_msg}")

    if content_manager and content_manager.collection:
        try:
            main_logger.info(f"\n获取内容集合 '{STD_CONTENT_COLLECTION_NAME}' 最终统计信息:")
            if not content_manager.is_loaded():
                main_logger.info(f"内容集合 '{STD_CONTENT_COLLECTION_NAME}' 未加载，尝试加载...")
                content_manager.load_collection()
            content_stats = content_manager.get_collection_stats()
            if content_stats:
                 main_logger.info(f"  最终实体数量: {content_stats['num_entities']}")
                 main_logger.info(f"  内存加载状态: {'是' if content_stats['is_loaded'] else '否'}")
            else: main_logger.warning("  无法获取内容集合的最终统计信息。")
        except Exception as e: main_logger.error(f"获取内容集合统计信息时出错: {e}", exc_info=True)

    if questions_manager and questions_manager.collection:
        try:
            main_logger.info(f"\n获取问题集合 '{STD_QUESTIONS_COLLECTION_NAME}' 最终统计信息:")
            if not questions_manager.is_loaded():
                main_logger.info(f"问题集合 '{STD_QUESTIONS_COLLECTION_NAME}' 未加载，尝试加载...")
                questions_manager.load_collection()
            q_stats = questions_manager.get_collection_stats()
            if q_stats:
                 main_logger.info(f"  最终实体数量: {q_stats['num_entities']}")
                 main_logger.info(f"  内存加载状态: {'是' if q_stats['is_loaded'] else '否'}")
            else: main_logger.warning("  无法获取问题集合的最终统计信息。")
        except Exception as e: main_logger.error(f"获取问题集合统计信息时出错: {e}", exc_info=True)

    main_logger.info("--- 正在清理资源 ---")
    if content_manager:
        content_manager.clear_memory_resources()
        content_manager.disconnect()
    if questions_manager:
        questions_manager.clear_memory_resources()
        questions_manager.disconnect()

    main_logger.info("脚本执行完毕。")