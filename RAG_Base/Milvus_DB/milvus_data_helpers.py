# data_processor.py

import json
import os
import time
import logging
import numpy as np
try:
    from config.config import (
        CONTENT_UNIQUE_ID_FIELD,      # 用于内容集合的唯一ID字段名
        QUESTIONS_UNIQUE_ID_FIELD     # 用于问题集合的唯一ID字段名
    )
except ImportError:
    logging.critical("无法从 config.config 导入 CONTENT_UNIQUE_ID_FIELD 或 QUESTIONS_UNIQUE_ID_FIELD。")
    CONTENT_UNIQUE_ID_FIELD = "chunk_id" # 不推荐，应修复导入
    QUESTIONS_UNIQUE_ID_FIELD = "question_unique_id"


module_logger = logging.getLogger(__name__)

def load_and_prepare_data_from_json(json_path, 
                                    embedding_model, 
                                    target_schema_type, 
                                    batch_size=32):
    """
    从 JSON 文件加载数据，生成嵌入向量，并为 Milvus 插入做准备。
    （您的函数代码保持不变）
    """
    module_logger.info(f"开始从 JSON 文件准备数据: {json_path}，目标 Schema 类型: {target_schema_type}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data_items = json.load(f)

        if not isinstance(raw_data_items, list):
            if isinstance(raw_data_items, dict) and "metadata" in raw_data_items:
                raw_data_items = [raw_data_items]
                module_logger.info(f"输入的 JSON 是单个有效对象，已封装为列表进行处理。")
            else:
                module_logger.error(f"JSON 文件顶层结构不是项目列表或单个可识别的项目对象: {json_path}")
                return [], np.array([])
    except FileNotFoundError:
        module_logger.error(f"文件未找到: {json_path}")
        return [], np.array([])
    except json.JSONDecodeError as e:
        module_logger.error(f"解析 JSON 文件失败: {json_path} - {e}", exc_info=True)
        return [], np.array([])
    except Exception as e:
        module_logger.error(f"加载 JSON 文件时发生未知错误: {json_path} - {e}", exc_info=True)
        return [], np.array([])

    if not raw_data_items:
        module_logger.warning(f"JSON 文件为空或解析后不包含任何数据项: {json_path}")
        return [], np.array([])

    module_logger.info(f"已从 {json_path} 加载 {len(raw_data_items)} 个原始项目。")

    texts_to_embed = []
    prepared_data_list = []
    source_filename = os.path.basename(json_path)
    valid_items_generated = 0
    original_items_processed_for_text = 0
    original_items_skipped = 0

    for i, item in enumerate(raw_data_items):
        if not isinstance(item, dict) or "metadata" not in item:
            module_logger.warning(f"跳过第 {i+1} 个原始项目: 格式无效或缺少 'metadata' 字段。项目内容 (前100字符): {str(item)[:100]}...")
            original_items_skipped += 1
            continue

        original_items_processed_for_text +=1
        metadata = item.get("metadata", {})
        content_text = str(item.get("content", "") or "")
        questions_text = str(item.get("questions", "") or "")

        chunk_id = metadata.get("chunk_id", f"未知块_{source_filename}_{i}")
        milvus_doc_title = metadata.get("originFileName", metadata.get("doc_title", "未知文档"))
        if not milvus_doc_title: milvus_doc_title = "未知文档"
        metadata_doc_title_val = metadata.get("doc_title", "")
        section_title = metadata.get("section_title", "")
        source_type = metadata.get("type", "text")
        try:
            char_count = int(metadata.get("char_count", 0))
        except (ValueError, TypeError):
            char_count = 0
            module_logger.warning(f"元数据中的 char_count ('{metadata.get('char_count')}') 无效，已设为0。Chunk ID: {chunk_id}")

        resource_id = str(metadata.get("resourceId", "") or "")
        dc_id = str(metadata.get("dcId", "") or "")
        resource_name = str(metadata.get("resourceName", "") or "")

        if target_schema_type == "content":
            if not content_text.strip():
                module_logger.warning(f"为 'content' Schema 跳过原始项目 {i+1}: 'content' 字段为空或仅包含空白。Chunk ID: {chunk_id}")
                original_items_skipped += 1
                original_items_processed_for_text -=1
                continue

            texts_to_embed.append(content_text)
            # 确保这里生成的字典键名与 CONTENT_UNIQUE_ID_FIELD 的值（如 "chunk_id"）匹配
            prepared_data_list.append({
                CONTENT_UNIQUE_ID_FIELD: chunk_id, # 使用常量作为键（如果 load_and_prepare 与 config 解耦，则用字面量 "chunk_id"）
                "doc_title": milvus_doc_title, "metadata_doc_title": metadata_doc_title_val,
                "section_title": section_title, "source_type": source_type, "char_count": char_count,
                "resourceId": resource_id, "dcId": dc_id, "resourceName": resource_name,
                "content": content_text, "raw_questions": questions_text, "source_file": source_filename
            })
            valid_items_generated += 1

        elif target_schema_type == "questions":
            if not questions_text.strip() or questions_text.startswith("[问题生成"):
                module_logger.debug(f"原始项目 {i+1} 没有有效 'questions' 文本。Chunk ID: {chunk_id}。Questions: '{questions_text[:100]}'")
                continue

            individual_questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().startswith("[问题生成")]

            if not individual_questions:
                module_logger.debug(f"原始项目 {i+1} 的 'questions' 字段 ('{questions_text[:100]}') 解析后未产生实际有效问题。Chunk ID: {chunk_id}")
                continue

            for q_idx, single_question_text in enumerate(individual_questions):
                if not single_question_text: continue

                texts_to_embed.append(single_question_text)
                # 确保这里生成的字典键名与 QUESTIONS_UNIQUE_ID_FIELD 的值（如 "question_unique_id"）匹配
                prepared_data_list.append({
                    QUESTIONS_UNIQUE_ID_FIELD: f"{chunk_id}_q{q_idx}", # 使用常量作为键
                    "chunk_id": chunk_id, # 这个也对应Schema字段
                    "doc_title": milvus_doc_title, "metadata_doc_title": metadata_doc_title_val,
                    "section_title": section_title, "source_type": source_type,
                    "resourceId": resource_id, "dcId": dc_id, "resourceName": resource_name,
                    "question": single_question_text, "source_file": source_filename
                })
                valid_items_generated += 1
        else:
            module_logger.error(f"无效的目标 Schema 类型: {target_schema_type}")
            return [], np.array([])

    # ... (函数其余部分，即嵌入向量生成逻辑，保持不变) ...
    if not texts_to_embed:
        module_logger.warning(f"未找到用于目标类型 '{target_schema_type}' 的可嵌入文本。总共尝试处理 {original_items_processed_for_text} 个原始项目，其中 {original_items_skipped} 个被跳过。")
        return [], np.array([])

    module_logger.info(f"已准备 {len(texts_to_embed)} 条文本用于嵌入 (来自 {valid_items_generated} 个有效 Milvus 条目，源自 {original_items_processed_for_text} 个被处理的原始项目，另有 {original_items_skipped} 个原始项目被跳过)。")

    all_embeddings_list = []
    try:
        if not hasattr(embedding_model, 'encode_documents') or not callable(embedding_model.encode_documents):
            raise AttributeError("embedding_model 对象缺少 'encode_documents' 方法或该方法不可调用。")

        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        module_logger.info(f"开始为 {len(texts_to_embed)} 条文本生成嵌入向量，共 {total_batches} 批次...")
        start_time_embed = time.time()

        for i_batch in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i_batch : i_batch + batch_size]
            current_batch_num = i_batch // batch_size + 1
            module_logger.debug(f"正在处理嵌入批次 {current_batch_num}/{total_batches} (大小: {len(batch_texts)})...")
            batch_start_time = time.time()
            batch_embeddings = embedding_model.encode_documents(batch_texts)
            batch_time = time.time() - batch_start_time
            module_logger.debug(f"批次 {current_batch_num} 完成 (耗时 {batch_time:.3f} 秒)")

            if isinstance(batch_embeddings, np.ndarray):
                all_embeddings_list.extend(batch_embeddings.tolist())
            elif isinstance(batch_embeddings, list):
                all_embeddings_list.extend([list(emb) if isinstance(emb, np.ndarray) else emb for emb in batch_embeddings])
            else:
                module_logger.warning(f"嵌入模型返回未知类型 {type(batch_embeddings)}，尝试转换为列表...")
                try:
                    converted_batch = [list(emb) if isinstance(emb, np.ndarray) else list(emb) if hasattr(emb, '__iter__') else [float(emb)] if isinstance(emb, (int, float)) else emb for emb in batch_embeddings]
                    all_embeddings_list.extend(converted_batch)
                except Exception as convert_e:
                    module_logger.error(f"无法将模型输出转换为嵌入列表: {convert_e}", exc_info=True)
                    raise TypeError("嵌入模型输出格式不兼容。")

        total_embed_time = time.time() - start_time_embed
        module_logger.info(f"嵌入向量生成完成 (总耗时 {total_embed_time:.2f} 秒)。共生成 {len(all_embeddings_list)} 个嵌入向量。")

        if not all_embeddings_list and texts_to_embed:
            module_logger.error("有待嵌入文本，但最终嵌入列表为空！嵌入模型可能未正确返回向量数据。")
            return prepared_data_list, np.array([])

        embeddings_array = np.array(all_embeddings_list, dtype='float32')

        if len(prepared_data_list) != embeddings_array.shape[0]:
            module_logger.critical(
                f"最终数据列表大小 ({len(prepared_data_list)}) 与最终嵌入向量数量 ({embeddings_array.shape[0]}) 不匹配！"
                f"Texts_to_embed 数量: {len(texts_to_embed)}."
            )
            return [], np.array([])
        return prepared_data_list, embeddings_array
    except AttributeError as ae:
        module_logger.error(f"嵌入模型配置错误: {ae}", exc_info=True)
        return [], np.array([])
    except TypeError as te:
        module_logger.error(f"嵌入模型输出格式不兼容或转换失败: {te}", exc_info=True)
        return [], np.array([])
    except Exception as e:
        module_logger.error(f"生成嵌入向量时发生严重错误: {e}", exc_info=True)
        return [], np.array([])


def add_new_json_to_milvus(
    json_file_path,
    content_milvus_manager,      # MilvusManager instance for content
    questions_milvus_manager,    # MilvusManager instance for questions
    embedding_model_instance,
    batch_size_embed=32
):
    """
    处理一个新的JSON文件，并将其数据添加到相应的Content和Questions Milvus集合中。
    包含去重逻辑。现在直接从 config.config 导入 UNIQUE_ID_FIELD 常量。
    """
    module_logger.info(f"\n--- 开始处理新JSON文件并添加到Milvus: {json_file_path} ---")

    # --- 1. 为内容集合处理和添加新数据 ---
    module_logger.info(f"--- 正在为内容集合准备来自 '{json_file_path}' 的数据 ---")
    prepared_content_data, content_embeddings = load_and_prepare_data_from_json(
        json_path=json_file_path,
        embedding_model=embedding_model_instance,
        target_schema_type="content",
        batch_size=batch_size_embed
    )

    if prepared_content_data and content_embeddings is not None and len(content_embeddings) > 0:
        module_logger.info(f"已为内容集合从新文件加载 {len(prepared_content_data)} 条数据。")
        
        existing_content_ids = set()
        if content_milvus_manager.collection and content_milvus_manager.collection.num_entities > 0:
            # 使用导入的 CONTENT_UNIQUE_ID_FIELD
            all_new_chunk_ids = [item.get(CONTENT_UNIQUE_ID_FIELD) for item in prepared_content_data if item.get(CONTENT_UNIQUE_ID_FIELD)]
            if all_new_chunk_ids:
                batch_size_dedup = 512
                for i in range(0, len(all_new_chunk_ids), batch_size_dedup):
                    batch_ids_to_check = all_new_chunk_ids[i : i + batch_size_dedup]
                    query_expr = f"{CONTENT_UNIQUE_ID_FIELD} in {json.dumps(batch_ids_to_check)}" # 使用导入的常量
                    try:
                        module_logger.debug(f"内容去重查询 (批次 {i//batch_size_dedup + 1}): {query_expr[:200]}...")
                        existing_docs = content_milvus_manager.query(
                            expr=query_expr,
                            output_fields=[CONTENT_UNIQUE_ID_FIELD], # 使用导入的常量
                            limit=len(batch_ids_to_check)
                        )
                        for doc in existing_docs:
                            existing_content_ids.add(doc[CONTENT_UNIQUE_ID_FIELD]) # 使用导入的常量
                    except Exception as e:
                        module_logger.warning(f"内容数据去重查询时发生错误 (批次): {type(e).__name__} - {str(e)[:100]}", exc_info=False)
                module_logger.info(f"在内容集合中找到 {len(existing_content_ids)} 个已存在的 ID (来自新文件中的候选项)。")

        content_to_add = []
        embeddings_for_content_add = []
        for idx, data_item in enumerate(prepared_content_data):
            item_id = data_item.get(CONTENT_UNIQUE_ID_FIELD) # 使用导入的常量
            if item_id and item_id not in existing_content_ids:
                content_to_add.append(data_item)
                embeddings_for_content_add.append(content_embeddings[idx])
            elif item_id:
                module_logger.info(f"跳过已存在于内容集合的 ID: {item_id}")
            else:
                module_logger.warning(f"内容数据项缺少唯一ID ('{CONTENT_UNIQUE_ID_FIELD}')，无法进行去重检查，将尝试添加: {str(data_item)[:100]}...")
                content_to_add.append(data_item)
                embeddings_for_content_add.append(content_embeddings[idx])

        if content_to_add:
            # ... (添加数据的逻辑不变) ...
            module_logger.info(f"准备向内容集合插入 {len(content_to_add)} 条新数据。")
            try:
                insert_result = content_milvus_manager.add_data(embeddings_for_content_add, content_to_add)
                if insert_result and insert_result.insert_count > 0:
                     module_logger.info(f"成功向内容集合添加 {insert_result.insert_count} 条新数据。")
                elif insert_result : # insert_count == 0
                     module_logger.warning(f"向内容集合添加数据时未插入任何条目，尽管尝试了添加。")
                else: # add_data returned None
                     module_logger.error(f"向内容集合添加新数据失败，add_data 返回 None。")

                if not content_milvus_manager.is_loaded():
                    content_milvus_manager.load_collection()
            except Exception as e:
                module_logger.error(f"向内容集合添加新数据时失败: {e}", exc_info=True)
        else:
            module_logger.info("去重后，没有新的内容数据可添加到内容集合。")
    else:
        module_logger.warning(f"从 '{json_file_path}' 加载或准备内容数据失败，或没有有效内容数据。")


    # --- 2. 为问题集合处理和添加新数据 ---
    module_logger.info(f"--- 正在为问题集合准备来自 '{json_file_path}' 的数据 ---")
    prepared_question_data, question_embeddings = load_and_prepare_data_from_json(
        json_path=json_file_path,
        embedding_model=embedding_model_instance,
        target_schema_type="questions",
        batch_size=batch_size_embed
    )

    if prepared_question_data and question_embeddings is not None and len(question_embeddings) > 0:
        module_logger.info(f"已为问题集合从新文件加载 {len(prepared_question_data)} 条数据。")

        existing_question_ids = set()
        if questions_milvus_manager.collection and questions_milvus_manager.collection.num_entities > 0:
            # 使用导入的 QUESTIONS_UNIQUE_ID_FIELD
            all_new_question_ids = [item.get(QUESTIONS_UNIQUE_ID_FIELD) for item in prepared_question_data if item.get(QUESTIONS_UNIQUE_ID_FIELD)]
            if all_new_question_ids:
                batch_size_dedup = 512
                for i in range(0, len(all_new_question_ids), batch_size_dedup):
                    batch_ids_to_check = all_new_question_ids[i : i + batch_size_dedup]
                    query_expr = f"{QUESTIONS_UNIQUE_ID_FIELD} in {json.dumps(batch_ids_to_check)}" # 使用导入的常量
                    try:
                        module_logger.debug(f"问题去重查询 (批次 {i//batch_size_dedup + 1}): {query_expr[:200]}...")
                        existing_q_docs = questions_milvus_manager.query(
                            expr=query_expr,
                            output_fields=[QUESTIONS_UNIQUE_ID_FIELD], # 使用导入的常量
                            limit=len(batch_ids_to_check)
                        )
                        for doc in existing_q_docs:
                            existing_question_ids.add(doc[QUESTIONS_UNIQUE_ID_FIELD]) # 使用导入的常量
                    except Exception as e:
                        module_logger.warning(f"问题数据去重查询时发生错误 (批次): {type(e).__name__} - {str(e)[:100]}", exc_info=False)
                module_logger.info(f"在问题集合中找到 {len(existing_question_ids)} 个已存在的 ID (来自新文件中的候选项)。")

        questions_to_add = []
        q_embeddings_for_add = []
        for idx, data_item in enumerate(prepared_question_data):
            item_id = data_item.get(QUESTIONS_UNIQUE_ID_FIELD) # 使用导入的常量
            if item_id and item_id not in existing_question_ids:
                questions_to_add.append(data_item)
                q_embeddings_for_add.append(question_embeddings[idx])
            elif item_id:
                module_logger.info(f"跳过已存在于问题集合的 ID: {item_id}")
            else:
                module_logger.warning(f"问题数据项缺少唯一ID ('{QUESTIONS_UNIQUE_ID_FIELD}')，无法去重，将尝试添加: {str(data_item)[:100]}...")
                questions_to_add.append(data_item)
                q_embeddings_for_add.append(question_embeddings[idx])

        if questions_to_add:
            # ... (添加数据的逻辑不变) ...
            module_logger.info(f"准备向问题集合插入 {len(questions_to_add)} 条新问题。")
            try:
                insert_q_result = questions_milvus_manager.add_data(q_embeddings_for_add, questions_to_add)
                if insert_q_result and insert_q_result.insert_count > 0:
                     module_logger.info(f"成功向问题集合添加 {insert_q_result.insert_count} 条新问题。")
                elif insert_q_result: # insert_count == 0
                     module_logger.warning(f"向问题集合添加数据时未插入任何条目，尽管尝试了添加。")
                else: # add_data returned None
                     module_logger.error(f"向问题集合添加新问题失败，add_data 返回 None。")

                if not questions_milvus_manager.is_loaded():
                    questions_milvus_manager.load_collection()
            except Exception as e:
                module_logger.error(f"向问题集合添加新问题时失败: {e}", exc_info=True)
        else:
            module_logger.info("去重后，没有新的问题数据可添加到问题集合。")
    else:
        module_logger.warning(f"从 '{json_file_path}' 加载或准备问题数据失败，或没有有效问题数据。")

    module_logger.info(f"--- 完成新JSON文件处理: {json_file_path} ---")