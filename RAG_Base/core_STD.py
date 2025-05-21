# main_api.py
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from fastapi.responses import StreamingResponse,JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager 
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
        STD_QUESTIONS_COLLECTION_NAME

    )
import config.config as app_config
from LLM_Response.llm_handler import LLMHandler
from Query.QueryProgessor import AdvancedQAQueryProcessor
from Embedding.embedding_generator import EmbeddingGenerator
from Milvus_DB.Milvus_Config import MilvusManager
from ReRank.Reranker import Reranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 全局变量，用于存储初始化后的对象 ---
# 这些将在应用启动时初始化
content_manager_global: MilvusManager = None
question_manager_global: MilvusManager = None
embedding_model_global: EmbeddingGenerator = None
processor_global: AdvancedQAQueryProcessor = None
reranker_instance_global: Reranker = None
use_reranker_flag_global: bool = False
llm_handler_instance_global: LLMHandler = None

logger = logging.getLogger("api") # 单独的API logger

def initialize_components():
    """初始化所有RAG组件并赋值给全局变量"""
    global content_manager_global, question_manager_global, embedding_model_global, \
           processor_global, reranker_instance_global, use_reranker_flag_global, \
           llm_handler_instance_global

    logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                        format=app_config.LOG_FORMAT)
    logger.info("开始初始化RAG组件...")

    try:
        content_manager_global = MilvusManager(
            dimension=app_config.DIMENSION, # 确保DIMENSION在config中定义
            collection_name=STD_CONTENT_COLLECTION_NAME,
            schema_fields_config=CONTENT_SCHEMA_FIELDS_CONFIG,
            schema_description=CONTENT_SCHEMA_DESCRIPTION,
            embedding_field_name=CONTENT_EMBEDDING_FIELD,
            text_field_name_for_bm25=CONTENT_TEXT_FIELD,
            tokenized_text_field_name_for_bm25=CONTENT_TOKENIZED_FIELD,
            unique_id_field_for_fusion=CONTENT_UNIQUE_ID_FIELD,
            host=app_config.MILVUS_HOST, port=app_config.MILVUS_PORT
        )
        logger.info("Content MilvusManager 初始化成功。")

        question_manager_global = MilvusManager(
            dimension=app_config.DIMENSION, # 确保DIMENSION在config中定义
            collection_name=STD_QUESTIONS_COLLECTION_NAME,
            schema_fields_config=QUESTIONS_SCHEMA_FIELDS_CONFIG,
            schema_description=QUESTIONS_SCHEMA_DESCRIPTION,
            embedding_field_name=QUESTIONS_EMBEDDING_FIELD,
            text_field_name_for_bm25=QUESTIONS_TEXT_FIELD,
            tokenized_text_field_name_for_bm25=QUESTIONS_TOKENIZED_FIELD,
            unique_id_field_for_fusion=QUESTIONS_UNIQUE_ID_FIELD,
            host=app_config.MILVUS_HOST, port=app_config.MILVUS_PORT
        )
        logger.info("Question MilvusManager 初始化成功。")

        embedding_model_global = EmbeddingGenerator(
            model_dir=app_config.EMBEDDING_MODEL_DIR,
            device=app_config.EMBEDDING_DEVICE,
            use_fp16=app_config.EMBEDDING_USE_FP16,
            query_instruction=app_config.EMBEDDING_QUERY_INSTRUCTION
        )
        logger.info("EmbeddingGenerator 初始化成功。")

        processor_global = AdvancedQAQueryProcessor(
            content_milvus_manager=content_manager_global,
            questions_milvus_manager=question_manager_global,
            embedding_model=embedding_model_global,
            default_top_k=app_config.INITIAL_SEARCH_TOP_K
        )
        logger.info("AdvancedQAQueryProcessor 初始化成功。")

        if app_config.USE_RERANKER_BY_DEFAULT:
            try:
                logger.info(f"正在加载 Reranker 模型: {app_config.RERANKER_MODEL_DIR} ({app_config.RERANKER_MODEL_TYPE}) 到设备: {app_config.RERANKER_DEVICE}")
                rerank_model_instance = AutoModelForSequenceClassification.from_pretrained(app_config.RERANKER_MODEL_DIR)
                rerank_tokenizer_instance = AutoTokenizer.from_pretrained(app_config.RERANKER_MODEL_DIR)
                reranker_instance_global = Reranker(
                    reranker_model=rerank_model_instance,
                    tokenizer=rerank_tokenizer_instance,
                    device=app_config.RERANKER_DEVICE
                )
                use_reranker_flag_global = True
                logger.info("Reranker 初始化成功。")
            except Exception as e:
                logger.error(f"Reranker 初始化失败: {e}。将不执行重排序。")
                reranker_instance_global = None
                use_reranker_flag_global = False
        else:
            logger.info("Reranker未配置为默认使用。")


        llm_handler_instance_global = LLMHandler(
            api_url=app_config.VLLM_URL,
            model_name=app_config.MODEL_NAME,
            system_prompt=app_config.STD_SYSTEM_PROMPT
        )
        logger.info("LLMHandler 初始化成功。")

    except Exception as e:
        logger.error(f"RAG组件初始化过程中发生严重错误: {e}", exc_info=True)
        # 在生产环境中，这里可能需要更优雅地处理，例如阻止应用启动或进入维护模式
        raise RuntimeError(f"RAG组件初始化失败: {e}")

    logger.info("所有RAG组件初始化完成。")

# FastAPI lifespan 事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    initialize_components()
    yield
    # 应用关闭时执行 (如果需要清理)
    logger.info("RAG应用正在关闭。")
    if content_manager_global:
        content_manager_global.close_connection() # 假设MilvusManager有close方法
    if question_manager_global:
        question_manager_global.close_connection()
    logger.info("Milvus连接已关闭。")


app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

async def process_rag_query(query_text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    处理RAG查询的核心逻辑，返回 (处理后的结果, 用于LLM的上下文)
    """
    if not processor_global or not llm_handler_instance_global:
        logger.error("RAG处理器或LLM处理器未初始化。")
        raise HTTPException(status_code=503, detail="服务组件未就绪，请稍后再试。")

    if not query_text.strip():
        logger.warning("接收到空查询。")
        # 返回空列表，让后续逻辑处理
        return [], []

    logger.info(f"开始处理查询: '{query_text[:100]}...'")
    initial_results = processor_global.process_query(query_text)

    if not initial_results:
        logger.info("初步搜索未返回任何结果。")
        return [], []

    final_processed_results = []
    if use_reranker_flag_global and reranker_instance_global:
        logger.info(f"准备对 {len(initial_results)} 条初步结果进行重排序...")
        results_for_reranking_input = []
        for i, entry in enumerate(initial_results):
            main_content_text = entry["content_details"].get(CONTENT_TEXT_FIELD, "")
            retrieved_question_objects_from_entry = entry.get("retrieved_questions", [])
            processed_question_texts = []
            if retrieved_question_objects_from_entry:
                for item_idx, item in enumerate(retrieved_question_objects_from_entry):
                    if isinstance(item, str): processed_question_texts.append(item)
                    elif isinstance(item, dict):
                        question_str = item.get(QUESTIONS_TEXT_FIELD, item.get("text", item.get("question", "")))
                        if question_str: processed_question_texts.append(question_str)
                        else: logger.warning(f"条目 {i}, Chunk ID {entry['chunk_id']}, 关联问题字典未能提取问题文本: {item}")
                    else: logger.warning(f"条目 {i}, Chunk ID {entry['chunk_id']}, 关联问题类型未知 ({type(item)}): {item}")

            combined_text_parts = [main_content_text]
            if processed_question_texts:
                combined_text_parts.append("\n\n--- 相关问题 ---")
                for q_text in processed_question_texts: combined_text_parts.append(q_text)
            text_for_reranker = "\n".join(filter(None, combined_text_parts)).strip()
            results_for_reranking_input.append({"Content": text_for_reranker, "original_entry": entry})

        reranked_list_with_original = reranker_instance_global.rerank(
            results=results_for_reranking_input, query=query_text,
            model_type=app_config.RERANKER_MODEL_TYPE, top_k=app_config.RERANK_TOP_K
        )
        logger.info(f"重排序完成，返回 {len(reranked_list_with_original)} 条结果。")

        for reranked_item in reranked_list_with_original:
            original_entry = reranked_item["original_entry"]
            final_entry = {
                "chunk_id": original_entry["chunk_id"],
                "final_score": reranked_item["Rerank Score"],
                "debug_scores": {**(original_entry.get("debug_scores", {})),
                                 "initial_fusion_score": original_entry.get("final_score")},
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
            valid_initial_results, key=lambda x: x.get("final_score", float('-inf')), reverse=True
        )[:app_config.DISPLAY_TOP_K_NO_RERANK]

    # 提取用于LLM的上下文信息 (仅包含文本内容和标题等关键信息)
    # 和用于前端展示的源文档详细信息
    contexts_for_llm = []
    source_documents_for_frontend = []

    num_contexts_to_llm = getattr(app_config, 'MAX_CONTEXTS_FOR_LLM_PROMPT', 5)

    for res_item in final_processed_results[:num_contexts_to_llm]: # 只选取配置数量的上下文给LLM
        content_details = res_item.get("content_details", {})
        contexts_for_llm.append({ # 这是传给 LLMHandler 的结构
            "chunk_id": res_item.get("chunk_id"),
            "content_details": {
                "content": content_details.get(CONTENT_TEXT_FIELD, "内容缺失"), # 确保是 content 字段
                "doc_title": content_details.get("doc_title", "未知文档"),
                "chunk_id": res_item.get("chunk_id") # LLMHandler 可能也需要
            }
        })
        source_documents_for_frontend.append({ 
            "chunk_id": res_item.get("chunk_id"),
            "title": content_details.get("doc_title", "未知文档"),
            "content": content_details.get(CONTENT_TEXT_FIELD, "内容缺失"),
            # "score": res_item.get("final_score"),
            "source_file": content_details.get("source_file"),
            "section_title": content_details.get("section_title"),
            "resourceId":content_details.get("resourceId"),
            "dcId":content_details.get("dcId"),
            "resourceName":content_details.get("dcId")
        })
    
    if not contexts_for_llm: # 如果 rerank 后结果为空
        logger.info("最终处理后没有可用的上下文。")

    return source_documents_for_frontend, contexts_for_llm


async def stream_rag_response(query_request: QueryRequest) -> AsyncGenerator[str, None]:
    """
    生成流式RAG响应：
    1. 源文档信息 (type="sources", data=...)
    2. LLM 回答块 (type="llm_chunk", token=...)
    3. 信息/错误消息 (type="info_message"/"error", data=...)
    4. LLM 流结束信号 (type="llm_done", token="[DONE]" 或其他标记)
    """
    source_documents = []
    contexts_for_llm = []
    llm_stream_completed = False # 标记LLM流是否已发送结束信号

    try:
        # 1. 处理查询获取源文档和上下文
        # process_rag_query 内部出错会抛出 HTTPException，由 endpoint 捕获
        source_documents, contexts_for_llm = await process_rag_query(query_request.query)

        # 2. 发送源文档信息
        source_data_event = {"type": "sources", "Reference": source_documents}
        yield f"data: {json.dumps(source_data_event, ensure_ascii=False)}\n\n"
        if source_documents:
            logger.info(f"已为查询 '{query_request.query[:50]}...' 发送 {len(source_documents)} 条源文档信息。")
        else:
            logger.info(f"查询 '{query_request.query[:50]}...' 未找到相关源文档。")

        # 3. 处理 LLM 回答或相应消息
        if contexts_for_llm and llm_handler_instance_global:
            logger.info(f"查询 '{query_request.query[:50]}...' 开始流式传输LLM回答，使用 {len(contexts_for_llm)} 个上下文。")
            async for llm_chunk in llm_handler_instance_global.generate_answer_stream_async(
                query=query_request.query,
                contexts=contexts_for_llm,
                max_tokens=getattr(app_config, 'LLM_DEFAULT_MAX_TOKENS', 16384),
                temperature=getattr(app_config, 'LLM_DEFAULT_TEMPERATURE', 0.1),
                max_contexts_for_llm=len(contexts_for_llm)
            ):
                # 检查 LLMHandler 是否返回了错误信息 (假设错误以特定前缀开始)
                if isinstance(llm_chunk, str) and llm_chunk.strip().startswith("错误："):
                     logger.warning(f"LLM handler yielded an error chunk: {llm_chunk}")
                     error_event = {"type": "error", "data": llm_chunk} # 错误消息用 data
                     yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                     # 可以在这里决定是否中断并发送错误类型的 llm_done
                elif llm_chunk: # 确保块不为空且不是错误
                    # *** 修改：使用 token 作为 key ***
                    llm_data_event = {"type": "llm_chunk", "token": llm_chunk}
                    yield f"data: {json.dumps(llm_data_event, ensure_ascii=False)}\n\n"

            # LLM 的 async for 循环正常结束后，发送结束信号
            # *** 修改：使用 token key 和 "[DONE]" 值 ***
            done_event = {"type": "llm_done", "token": "[DONE]"}
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
            llm_stream_completed = True
            logger.info(f"已为查询 '{query_request.query[:50]}...' 发送LLM结束标记 [DONE]。")

        elif not contexts_for_llm:
            logger.info(f"查询 '{query_request.query[:50]}...' 没有足够上下文，无法生成LLM回答。")
            # 发送提示信息，使用 data key
            info_message = {"type": "info_message", "data": "抱歉，未能找到足够的相关信息来回答您的问题。"}
            yield f"data: {json.dumps(info_message, ensure_ascii=False)}\n\n"
            # 即使没有 LLM 输出，也发送一个完成信号表明 LLM 处理环节结束
            done_event_no_context = {"type": "llm_done", "token": "[NO_CONTEXT]"} # 或使用 "[DONE]"
            yield f"data: {json.dumps(done_event_no_context, ensure_ascii=False)}\n\n"
            llm_stream_completed = True

        else: # llm_handler_instance_global is None
            logger.error("LLM处理器未初始化，无法生成回答。")
            # 发送错误信息，使用 data key
            error_event = {"type": "error", "data": "LLM服务当前不可用，请稍后再试。"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            # 发送一个完成信号表明 LLM 处理环节因错误结束
            done_event_error = {"type": "llm_done", "token": "[ERROR_HANDLER]"} # 或使用 "[DONE]"
            yield f"data: {json.dumps(done_event_error, ensure_ascii=False)}\n\n"
            llm_stream_completed = True

    except HTTPException as http_exc:
        # 这个异常通常在 process_rag_query 中抛出，会在 endpoint 级别被处理
        # 但如果流已经开始后才发生（理论上不应该），这里记录一下
        logger.error(f"流处理中发生 HTTPException: {http_exc.detail}", exc_info=True)
        # 尝试发送错误事件
        try:
            error_event = {"type": "error", "data": f"处理错误: {http_exc.detail}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            # 如果LLM流还没结束，发送一个错误结束信号
            if not llm_stream_completed:
                done_event_http_error = {"type": "llm_done", "token": "[ERROR_HTTP]"}
                yield f"data: {json.dumps(done_event_http_error, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"发送 HTTPException 的错误事件失败: {e}")
        # 重新抛出，让 FastAPI 处理
        raise http_exc
    except Exception as e:
        logger.error(f"流式响应生成期间发生意外错误: {e}", exc_info=True)
        # 尝试发送最终错误事件
        try:
            error_event = {"type": "error", "data": f"服务器内部流处理错误: {str(e)}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            # 如果LLM流还没结束，发送一个错误结束信号
            if not llm_stream_completed:
                done_event_unknown_error = {"type": "llm_done", "token": "[ERROR_UNKNOWN]"}
                yield f"data: {json.dumps(done_event_unknown_error, ensure_ascii=False)}\n\n"
        except Exception as inner_e:
            logger.error(f"发送最终错误事件失败: {inner_e}")
    finally:
        # 可选：如果需要一个全局的流结束信号（独立于 llm_done）
        # final_rag_end_event = {"type": "rag_stream_end", "data": {"status": "finished"}}
        # yield f"data: {json.dumps(final_rag_end_event, ensure_ascii=False)}\n\n"
        logger.debug(f"查询 '{query_request.query[:50]}...' 的 stream_rag_response 生成器结束。")


@app.post("///", summary="处理RAG查询并流式返回结果")
async def query_stream_endpoint(request_body: QueryRequest):
    """
    接收用户查询，执行RAG流程，并以Server-Sent Events (SSE) 的形式流式返回：
    1. 源文档信息 (type="sources", data=...)
    2. LLM生成的回答文本块 (type="llm_chunk", token=...)
    3. 信息/错误消息 (type="info_message"/"error", data=...)
    4. LLM流结束信号 (type="llm_done", token="[DONE]" 或其他标记)
    """
    try:
        # 设置 SSE 相关的 Headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # 对 Nginx 等反向代理重要
        }
        return StreamingResponse(stream_rag_response(request_body), headers=headers)
    except HTTPException as http_exc:
        # 捕获在 stream_rag_response 开始前（主要是在 process_rag_query 中）抛出的 HTTP 异常
        logger.warning(f"请求处理失败 (HTTPException): {http_exc.status_code} - {http_exc.detail}")
        # 返回标准的 JSON 错误响应，因为流还没有开始
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"detail": http_exc.detail},
        )
    except Exception as e:
        # 捕获在 stream_rag_response 开始前发生的其他意外错误
        logger.error(f"处理流式查询端点时发生意外错误: {e}", exc_info=True)
        # 返回标准的 JSON 错误响应
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务器内部错误: {str(e)}"},
        )

if __name__ == "__main__":
    import uvicorn
    # 确保在Config/config.py中定义了DIMENSION
    if not hasattr(app_config, 'DIMENSION'):
        print("错误: Config/config.py 文件中缺少 'DIMENSION' 配置项。")
        exit(1)
    uvicorn.run("core_STD:app", host=app_config.SERVER_HOST, port=app_config.SERVER_PORT, reload=False) # reload=False for direct run

# curl -X POST \
#      -H "Content-Type: application/json" \
#      -H "Accept: text/event-stream" \
#      -d '{"query": "省级国土空间规划有哪些流程"}' \
#      --no-buffer \
#      http://192.168.1.240:5003/CdipdRAGSTD/Standard/Query
