import json
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config.config as app_config # 使用别名
from LLM_Response.llm_handler import LLMHandler 

from Query.QueryProgessor import AdvancedQAQueryProcessor
from Embedding.embedding_generator import EmbeddingGenerator
from Milvus_DB.Milvus_Config import MilvusManager
from ReRank.Reranker import Reranker
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

        STD_CONTENT_COLLECTION_NAME,
        STD_QUESTIONS_COLLECTION_NAME,

        MILVUS_HOST,
        MILVUS_PORT    
    )

def main():
    # --- 初始化日志 ---
    logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),format=app_config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    # --- 初始化 MilvusManager 和 EmbeddingGenerator ---
    content_manager = MilvusManager(
    dimension=app_config.DIMENSION, collection_name=STD_CONTENT_COLLECTION_NAME,
    schema_fields_config=CONTENT_SCHEMA_FIELDS_CONFIG, schema_description=CONTENT_SCHEMA_DESCRIPTION,
    embedding_field_name=CONTENT_EMBEDDING_FIELD, text_field_name_for_bm25=CONTENT_TEXT_FIELD,
    tokenized_text_field_name_for_bm25=CONTENT_TOKENIZED_FIELD, unique_id_field_for_fusion=CONTENT_UNIQUE_ID_FIELD,
    host=MILVUS_HOST, port=MILVUS_PORT
)

    question_manager = MilvusManager(
    dimension=app_config.DIMENSION, collection_name=STD_QUESTIONS_COLLECTION_NAME,
    schema_fields_config=QUESTIONS_SCHEMA_FIELDS_CONFIG, schema_description=QUESTIONS_SCHEMA_DESCRIPTION,
    embedding_field_name=QUESTIONS_EMBEDDING_FIELD, text_field_name_for_bm25=QUESTIONS_TEXT_FIELD,
    tokenized_text_field_name_for_bm25=QUESTIONS_TOKENIZED_FIELD, unique_id_field_for_fusion=QUESTIONS_UNIQUE_ID_FIELD,
    host=MILVUS_HOST, port=MILVUS_PORT
)

    embedding_model = EmbeddingGenerator(
        model_dir=app_config.EMBEDDING_MODEL_DIR,
        device=app_config.EMBEDDING_DEVICE,
        use_fp16=app_config.EMBEDDING_USE_FP16,
        query_instruction=app_config.EMBEDDING_QUERY_INSTRUCTION
    )

    # --- 初始化 AdvancedQAQueryProcessor ---
    processor = AdvancedQAQueryProcessor(
        content_milvus_manager=content_manager,
        questions_milvus_manager=question_manager,
        embedding_model=embedding_model,
        default_top_k=app_config.INITIAL_SEARCH_TOP_K
    )

    # --- 初始化 Reranker ---
    reranker_instance = None
    use_reranker_flag = False
    if app_config.USE_RERANKER_BY_DEFAULT:
        try:
            logger.info(f"正在加载 Reranker 模型: {app_config.RERANKER_MODEL_DIR} ({app_config.RERANKER_MODEL_TYPE}) 到设备: {app_config.RERANKER_DEVICE}")
            rerank_model_instance = AutoModelForSequenceClassification.from_pretrained(app_config.RERANKER_MODEL_DIR)
            rerank_tokenizer_instance = AutoTokenizer.from_pretrained(app_config.RERANKER_MODEL_DIR)

            reranker_instance = Reranker(
                reranker_model=rerank_model_instance,
                tokenizer=rerank_tokenizer_instance,
                device=app_config.RERANKER_DEVICE
            )
            use_reranker_flag = True
            logger.info("Reranker 初始化成功。")
        except Exception as e:
            logger.error(f"Reranker 初始化失败: {e}。将不执行重排序。")
            # reranker_instance 保持为 None, use_reranker_flag 保持为 False

    # --- 初始化 LLMHandler ---
    llm_handler_instance = None
    try:
        llm_handler_instance = LLMHandler(
            api_url=app_config.VLLM_URL,
            model_name=app_config.MODEL_NAME,
            system_prompt=app_config.STD_SYSTEM_PROMPT
        )
        logger.info("LLMHandler 初始化成功。")
    except AttributeError as e: # 捕获 config 文件中可能缺少的属性
        logger.error(f"LLMHandler 初始化失败：确保 VLLM_URL 和 MODEL_NAME 在配置文件中已定义。错误: {e}")
    except Exception as e:
        logger.error(f"LLMHandler 初始化失败: {e}。将无法生成 LLM 回答。")


    # --- 获取用户查询 ---
    query_text = input("请输入查询问题: ")
    if not query_text.strip():
        logger.error("查询不能为空。")
        return

    # --- 执行初步的多阶段混合搜索 ---
    initial_results = processor.process_query(query_text)

    if not initial_results:
        logger.info("初步搜索未返回任何结果。")
        print("查询结果:\n未找到相关结果。")
        if llm_handler_instance:
            print("\n--- LLM 生成的回答 ---")
            print("无法生成回答，因为没有找到相关的上下文信息。")
        return

    final_processed_results = []

    if use_reranker_flag and reranker_instance:
        logger.info(f"准备对 {len(initial_results)} 条初步结果进行重排序...")
        
        results_for_reranking_input = []
        for i, entry in enumerate(initial_results):
            main_content_text = entry["content_details"].get(CONTENT_TEXT_FIELD, "")   
            retrieved_question_objects_from_entry = entry.get("retrieved_questions", [])
            processed_question_texts = []
            if retrieved_question_objects_from_entry:
                for item_idx, item in enumerate(retrieved_question_objects_from_entry):
                    if isinstance(item, str):
                        processed_question_texts.append(item)
                    elif isinstance(item, dict):
                        question_str = item.get(QUESTIONS_TEXT_FIELD, item.get("text", item.get("question", "")))
                        if question_str:
                            processed_question_texts.append(question_str)
                        else:
                            logger.warning(f"条目 {i}, Chunk ID {entry['chunk_id']}, 关联问题字典未能提取问题文本: {item}")
                    else:
                        logger.warning(f"条目 {i}, Chunk ID {entry['chunk_id']}, 关联问题类型未知 ({type(item)}): {item}")

            combined_text_parts = [main_content_text]
            if processed_question_texts:
                combined_text_parts.append("\n\n--- 相关问题 ---")
                for q_text in processed_question_texts:
                    combined_text_parts.append(q_text)
            text_for_reranker = "\n".join(filter(None, combined_text_parts)).strip()
            results_for_reranking_input.append({
                "Content": text_for_reranker,
                "original_entry": entry
            })

        reranked_list_with_original = reranker_instance.rerank(
            results=results_for_reranking_input,
            query=query_text,
            model_type=app_config.RERANKER_MODEL_TYPE,
            top_k=app_config.RERANK_TOP_K # 使用配置
        )
        logger.info(f"重排序完成，返回 {len(reranked_list_with_original)} 条结果。")

        for reranked_item in reranked_list_with_original:
            original_entry = reranked_item["original_entry"]
            final_entry = {
                "chunk_id": original_entry["chunk_id"],
                "final_score": reranked_item["Rerank Score"],
                "debug_scores": {
                    **(original_entry.get("debug_scores", {})), # 安全访问
                    "initial_fusion_score": original_entry.get("final_score") # 安全访问
                },
                "content_details": original_entry["content_details"],
                "retrieved_questions": original_entry.get("retrieved_questions", []),
            }
            final_processed_results.append(final_entry)
    else:
        logger.info("未执行重排序，使用初步搜索结果。")
        valid_initial_results = [res for res in initial_results if isinstance(res.get("final_score"), (int, float))]
        if len(valid_initial_results) < len(initial_results):
            logger.warning("部分初步结果的 final_score 无效，已被过滤。")

        final_processed_results = sorted(
            valid_initial_results,
            key=lambda x: x.get("final_score", float('-inf')),
            reverse=True
        )[:app_config.DISPLAY_TOP_K_NO_RERANK] # 使用配置


    print("\n--- RAG 检索与重排结果 (传递给LLM的上下文) ---")
    if final_processed_results:
        # 打印少量结果用于调试，避免控制台输出过多
        print(json.dumps(final_processed_results[:min(3, len(final_processed_results))], ensure_ascii=False, indent=2))
        if len(final_processed_results) > 3:
            print(f"...及另外 {len(final_processed_results) - 3} 条结果未显示。")
    else:
        print("未找到相关结果。")
        if llm_handler_instance:
            print("\n--- LLM 生成的回答 ---")
            print("无法生成回答，因为没有找到相关的上下文信息。")
        return # 如果没有结果，则不调用LLM

    # --- 调用 LLM 生成回答 ---
    if final_processed_results and llm_handler_instance:
        logger.info("获取 LLM 回答...")
        print("\n--- LLM 生成的回答 ---")
        llm_handler_instance.generate_answer_stream( 
            query=query_text,
            contexts=final_processed_results,
            max_tokens=getattr(app_config, 'LLM_DEFAULT_MAX_TOKENS', 16384),
            temperature=getattr(app_config, 'LLM_DEFAULT_TEMPERATURE', 0.6),
            max_contexts_for_llm=getattr(app_config, 'MAX_CONTEXTS_FOR_LLM_PROMPT', 10)
        )
    elif not llm_handler_instance:
        print("\n--- LLM 生成的回答 ---")
        print("LLM 处理器未成功初始化，无法生成回答。")
    else: # final_processed_results 为空的情况已在上面处理
        pass


if __name__ == "__main__":
    main()