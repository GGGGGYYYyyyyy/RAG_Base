# query/qa_processor.py
# -*- coding: utf-8 -*-

import numpy as np
import torch
import time
import logging
from typing import List, Dict, Any, Union, Optional

# --- 导入依赖 ---

from Embedding.embedding_generator import EmbeddingGenerator
from Milvus_DB.Milvus_Config import MilvusManager
from config.config import (
    CONTENT_SCHEMA_FIELDS_CONFIG, 
    CONTENT_EMBEDDING_FIELD, 
    CONTENT_TEXT_FIELD, 
    CONTENT_TOKENIZED_FIELD, 
    CONTENT_UNIQUE_ID_FIELD,
                        
    QUESTIONS_SCHEMA_FIELDS_CONFIG,
    QUESTIONS_EMBEDDING_FIELD,
    QUESTIONS_TEXT_FIELD,
    QUESTIONS_TOKENIZED_FIELD, 
    QUESTIONS_UNIQUE_ID_FIELD
    )



class AdvancedQAQueryProcessor:
    """
    高级查询处理器，执行多阶段、跨集合的QA查询（暂时不含ReRank）。
    """
    def __init__(self,
                 content_milvus_manager: MilvusManager,
                 questions_milvus_manager: MilvusManager,
                 embedding_model: Optional[EmbeddingGenerator],
                 default_top_k: int = 5,
                 content_search_top_k_internal: int = 30,
                 question_search_top_k_internal: int = 30,
                 final_fusion_rrf_k: int = 60):
        """
        初始化高级 QA 查询处理器。

        Args:
            content_milvus_manager: 内容集合的 MilvusManager 实例。
            questions_milvus_manager: 问题集合的 MilvusManager 实例。
            embedding_model: 用于生成查询向量的嵌入模型 (可选, 但混合搜索需要)。
            default_top_k: 默认最终返回的结果数量。
            content_search_top_k_internal: 内容搜索阶段召回的候选数量。
            question_search_top_k_internal: 问题搜索阶段召回的候选数量。
            final_fusion_rrf_k: 最终跨集合融合时的RRF k参数。
        """
        if not isinstance(content_milvus_manager, MilvusManager):
            raise TypeError("content_milvus_manager 必须是 MilvusManager 的实例")
        if not isinstance(questions_milvus_manager, MilvusManager):
            raise TypeError("questions_milvus_manager 必须是 MilvusManager 的实例")
        if embedding_model is not None and not hasattr(embedding_model, 'encode_documents'):
             raise TypeError("embedding_model 必须是类似 EmbeddingGenerator 的实例或 None")

        self.content_manager = content_milvus_manager
        self.questions_manager = questions_milvus_manager
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k
        self.content_search_top_k_internal = content_search_top_k_internal
        self.question_search_top_k_internal = question_search_top_k_internal
        self.final_fusion_rrf_k = final_fusion_rrf_k

        self.logger = logging.getLogger("高级QA查询处理器")
        self.logger.info(f"高级QA查询处理器初始化完成。")

    def _generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """辅助函数：生成查询嵌入。"""
        if not self.embedding_model:
            self.logger.error("需要嵌入模型来生成查询嵌入，但初始化时未提供。")
            return None
        try:
            self.logger.info("正在为查询生成嵌入向量...")
            start_time = time.time()
            q_embedding_data = self.embedding_model.encode_documents([query_text], batch_size=1)
            query_embedding_flat: Optional[List[float]] = None
            
            # 处理嵌入模型的各种可能输出格式
            if isinstance(q_embedding_data, np.ndarray):
                if q_embedding_data.ndim == 1: 
                    query_embedding_flat = q_embedding_data.tolist()
                elif q_embedding_data.ndim == 2 and q_embedding_data.shape[0] == 1:
                    query_embedding_flat = q_embedding_data[0].tolist()
                else:
                    self.logger.warning(f"嵌入模型返回了非预期的Numpy数组形状: {q_embedding_data.shape}")
                    if q_embedding_data.size > 0: 
                         # 尝试扁平化并取预期维度
                         query_embedding_flat = q_embedding_data.reshape(-1).tolist()
                         if len(query_embedding_flat) > self.content_manager.dimension:
                             query_embedding_flat = query_embedding_flat[:self.content_manager.dimension]
                         # 可以添加维度不匹配的日志
            elif isinstance(q_embedding_data, list) and q_embedding_data:
                if isinstance(q_embedding_data[0], (list, np.ndarray)): # 列表的列表/数组
                     query_embedding_flat = np.array(q_embedding_data[0]).flatten().tolist()
                elif isinstance(q_embedding_data[0], (int, float)): # 扁平列表
                     query_embedding_flat = q_embedding_data
                else: 
                    raise TypeError(f"从 embedding_generator 接收到意外的嵌入列表元素类型: {type(q_embedding_data[0])}")
            else: 
                raise TypeError(f"从 embedding_generator 接收到意外的嵌入类型: {type(q_embedding_data)}")
            
            # 维度检查
            expected_dim = self.content_manager.dimension # 假设内容和问题集合维度一致
            if query_embedding_flat and len(query_embedding_flat) != expected_dim:
                self.logger.error(f"生成的查询嵌入维度 ({len(query_embedding_flat)}) 与集合维度 ({expected_dim}) 不匹配！")
                return None

            embed_time = time.time() - start_time
            self.logger.info(f"查询嵌入向量生成完成 (耗时 {embed_time:.3f} 秒)")
            return query_embedding_flat
        except Exception as e:
            self.logger.error(f"生成查询嵌入向量失败: {e}", exc_info=True)
            return None

    def multi_stage_hybrid_search(self,
                                  query_text: str,
                                  top_k: Optional[int] = None,
                                  filters_expr_content: Optional[str] = None,
                                  filters_expr_question: Optional[str] = None,
                                  content_vector_w: float = 0.6, content_bm25_w: float = 0.4,
                                  question_vector_w: float = 0.6, question_bm25_w: float = 0.4
                                 ) -> List[Dict[str, Any]]:
        """执行多阶段、跨集合的 RRF 混合搜索 (无 ReRank)。"""
        if top_k is None:
            top_k = self.default_top_k

        query_embedding = self._generate_query_embedding(query_text)

        # --- 阶段一: 内容集合混合检索 ---
        self.logger.info(f"--- 多阶段搜索阶段 1: 内容集合 '{self.content_manager.collection_name}' 混合检索 ---")
        content_results_from_manager: List[Dict] = []
        try:
            # 确定内容集合的输出字段
            content_output_fields = [
                f['name'] for f in CONTENT_SCHEMA_FIELDS_CONFIG 
                if f['name'] not in [CONTENT_EMBEDDING_FIELD, CONTENT_TOKENIZED_FIELD]
            ]
            # 确保唯一ID字段被包含
            if CONTENT_UNIQUE_ID_FIELD not in content_output_fields: 
                content_output_fields.append(CONTENT_UNIQUE_ID_FIELD)
            
            content_results_from_manager = self.content_manager.hybrid_search(
                query_embeddings=query_embedding, query_text=query_text,
                top_k=self.content_search_top_k_internal, expr=filters_expr_content,
                fusion_method="rrf", 
                vector_weight=content_vector_w, bm25_weight=content_bm25_w,
                output_fields=list(set(content_output_fields)) 
            )
            self.logger.info(f"内容集合检索到 {len(content_results_from_manager)} 个候选结果。")
        except Exception as e:
            self.logger.error(f"内容集合混合搜索失败: {e}", exc_info=True)

        # --- 阶段二: 问题集合混合检索 ---
        self.logger.info(f"--- 多阶段搜索阶段 2: 问题集合 '{self.questions_manager.collection_name}' 混合检索 ---")
        question_results_from_manager: List[Dict] = []
        try:
            # 确定问题集合的输出字段
            question_output_fields = [
                f['name'] for f in QUESTIONS_SCHEMA_FIELDS_CONFIG
                if f['name'] not in [QUESTIONS_EMBEDDING_FIELD, QUESTIONS_TOKENIZED_FIELD]
            ]
            # 确保唯一问题ID和关联的chunk_id被包含
            if QUESTIONS_UNIQUE_ID_FIELD not in question_output_fields:
                question_output_fields.append(QUESTIONS_UNIQUE_ID_FIELD)
            if "chunk_id" not in question_output_fields: # 直接使用 "chunk_id"
                question_output_fields.append("chunk_id") 
            
            question_results_from_manager = self.questions_manager.hybrid_search(
                query_embeddings=query_embedding, query_text=query_text,
                top_k=self.question_search_top_k_internal, expr=filters_expr_question,
                fusion_method="rrf", 
                vector_weight=question_vector_w, bm25_weight=question_bm25_w,
                output_fields=list(set(question_output_fields))
            )
            self.logger.info(f"问题集合检索到 {len(question_results_from_manager)} 个候选结果。")
        except Exception as e:
            self.logger.error(f"问题集合混合搜索失败: {e}", exc_info=True)

        # --- 阶段三: 跨集合结果的二次 RRF 融合 ---
        self.logger.info(f"--- 多阶段搜索阶段 3: 跨集合结果二次 RRF 融合 (基于 chunk_id, k={self.final_fusion_rrf_k}) ---")
        final_rrf_scores: Dict[str, Dict[str, Any]] = {}
        k_param_final_fusion = self.final_fusion_rrf_k

        # 1. 处理内容搜索结果
        for rank_c, content_item_dict in enumerate(content_results_from_manager):
            chunk_id = content_item_dict.get(CONTENT_UNIQUE_ID_FIELD)
            if not chunk_id: 
                self.logger.warning(f"内容结果项在排名 {rank_c} 处缺少 '{CONTENT_UNIQUE_ID_FIELD}'，跳过。")
                continue
            rrf_contribution = 1.0 / (k_param_final_fusion + rank_c + 1)
            if chunk_id not in final_rrf_scores:
                # 初始化条目结构
                final_rrf_scores[chunk_id] = {
                    "rrf_score": 0.0, 
                    "content_item_details": None, 
                    "content_item_rank": -1, 
                    "associated_question_items": [], 
                    "question_items_ranks": []
                }
            final_rrf_scores[chunk_id]["content_item_details"] = content_item_dict
            final_rrf_scores[chunk_id]["content_item_rank"] = rank_c
            final_rrf_scores[chunk_id]["rrf_score"] += rrf_contribution

        # 2. 处理问题搜索结果
        for rank_q, question_item_dict in enumerate(question_results_from_manager):
            chunk_id = question_item_dict.get("chunk_id") # 直接使用字段名 "chunk_id"
            if not chunk_id: 
                q_uid = question_item_dict.get(QUESTIONS_UNIQUE_ID_FIELD, "N/A")
                self.logger.warning(f"问题结果项 (Q_ID: {q_uid}, 排名 {rank_q}) 缺少关联的 'chunk_id'，跳过。")
                continue
            rrf_contribution = 1.0 / (k_param_final_fusion + rank_q + 1)
            if chunk_id not in final_rrf_scores:
                # 如果 chunk_id 仅在问题搜索中出现，也创建条目
                final_rrf_scores[chunk_id] = {
                    "rrf_score": 0.0, 
                    "content_item_details": None, # 内容详情待后续填充
                    "content_item_rank": -1,      # 标记内容搜索未命中
                    "associated_question_items": [], 
                    "question_items_ranks": []
                }
            final_rrf_scores[chunk_id]["rrf_score"] += rrf_contribution
            final_rrf_scores[chunk_id]["associated_question_items"].append(question_item_dict)
            final_rrf_scores[chunk_id]["question_items_ranks"].append(rank_q)

        # 按最终 RRF 分数排序
        sorted_chunk_data_with_scores = sorted(final_rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        self.logger.info(f"二次 RRF 融合后，得到 {len(sorted_chunk_data_with_scores)} 个唯一的 chunk_id 候选。")

        # --- 最终格式化输出 (无ReRank) ---
        final_output_list: List[Dict[str, Any]] = []
        # 只处理最终 top_k 个
        for i, chunk_data in enumerate(sorted_chunk_data_with_scores[:top_k]):
            current_chunk_id: Optional[str] = None
            content_details_for_item = chunk_data.get("content_item_details")

            # 确定 current_chunk_id
            if content_details_for_item and content_details_for_item.get(CONTENT_UNIQUE_ID_FIELD):
                current_chunk_id = content_details_for_item.get(CONTENT_UNIQUE_ID_FIELD)
            elif chunk_data.get("associated_question_items"): # 如果内容为空，但有关联问题
                # 确保列表非空
                if chunk_data["associated_question_items"]:
                    first_q_item = chunk_data["associated_question_items"][0]
                    current_chunk_id = first_q_item.get("chunk_id") # 直接用 "chunk_id"
            
            if not current_chunk_id:
                self.logger.warning(f"无法为融合结果项 (RRF排名 {i}) 确定 chunk_id，跳过。")
                continue

            # 如果 content_item_details 为 None (仅问题命中)，则尝试获取内容
            if content_details_for_item is None:
                self.logger.debug(f"Chunk ID '{current_chunk_id}' 内容详情缺失，从内容库获取...")
                try:
                    # 确定需要获取的内容字段
                    content_output_fields_ref = [f['name'] for f in CONTENT_SCHEMA_FIELDS_CONFIG if f['name'] not in [CONTENT_EMBEDDING_FIELD, CONTENT_TOKENIZED_FIELD]]
                    if CONTENT_UNIQUE_ID_FIELD not in content_output_fields_ref: 
                        content_output_fields_ref.append(CONTENT_UNIQUE_ID_FIELD)
                    
                    # 对 current_chunk_id 进行简单的引号转义
                    escaped_current_chunk_id = str(current_chunk_id).replace("'", "''")
                    retrieved_content_list = self.content_manager.query(
                        expr=f"{CONTENT_UNIQUE_ID_FIELD} == '{escaped_current_chunk_id}'",
                        output_fields=list(set(content_output_fields_ref)), limit=1
                    )
                    if retrieved_content_list:
                        content_details_for_item = retrieved_content_list[0]
                        self.logger.debug(f"成功获取 Chunk ID '{current_chunk_id}' 内容详情。")
                    else:
                        self.logger.warning(f"无法为 Chunk ID '{current_chunk_id}' 找到内容详情。")
                        # 创建一个包含错误信息的占位符
                        content_details_for_item = {CONTENT_UNIQUE_ID_FIELD: current_chunk_id, 'error': '内容未找到'}
                except Exception as e_fetch:
                    self.logger.error(f"为 Chunk ID '{current_chunk_id}' 获取内容详情时出错: {e_fetch}", exc_info=False)
                    content_details_for_item = {CONTENT_UNIQUE_ID_FIELD: current_chunk_id, 'error': f'获取内容出错: {e_fetch}'}
            
            # 对关联问题按其在问题搜索阶段的得分排序
            # 假设 MilvusManager.hybrid_search 返回的字典包含 'score'
            sorted_associated_questions = sorted(
                chunk_data.get('associated_question_items', []),
                key=lambda q: q.get('score', 0.0), 
                reverse=True
            )

            # 构建最终返回给用户的条目
            result_entry = {
                "chunk_id": current_chunk_id,
                "final_score": chunk_data.get('rrf_score', 0.0), # 二次RRF融合得分
                "debug_scores": { # 用于调试分析的额外分数信息
                    "content_search_rank": chunk_data.get('content_item_rank', -1), # 内容搜索排名 (-1表示未命中)
                    # 可以添加更多调试信息，如关联问题的最高/平均排名等
                },
                "content_details": content_details_for_item, # 完整的内容信息字典
                "retrieved_questions": sorted_associated_questions[:3] # 最多返回3个最相关的关联问题
            }
            final_output_list.append(result_entry)

        self.logger.info(f"多阶段混合搜索完成 (无ReRank)，最终返回 {len(final_output_list)} 个格式化结果。")
        return final_output_list


    def process_query(self,
                      query_text: str,
                      search_method: str = "multi_stage_hybrid",
                      top_k: Optional[int] = None,
                      filters_expr: Optional[str] = None, 
                      vector_weight: float = 0.6, 
                      bm25_weight: float = 0.4,
                      filters_expr_content: Optional[str] = None,
                      filters_expr_question: Optional[str] = None,
                      content_vector_w: float = 0.6, content_bm25_w: float = 0.4,
                      question_vector_w: float = 0.6, question_bm25_w: float = 0.4
                     ) -> List[Dict]:
        """
        处理用户查询，根据指定方法执行搜索并返回格式化的结果列表。
        """                     
        if top_k is None: top_k = self.default_top_k

        if search_method == "multi_stage_hybrid":
            # 调用多阶段搜索方法
            return self.multi_stage_hybrid_search(
                query_text, top_k,
                filters_expr_content if filters_expr_content is not None else filters_expr,
                filters_expr_question if filters_expr_question is not None else filters_expr,
                content_vector_w, content_bm25_w,
                question_vector_w, question_bm25_w
            )
        else: # 处理单阶段搜索逻辑 (例如，只搜索内容集合)
            self.logger.info(f"执行单阶段搜索 '{search_method}' (默认作用于内容集合)...")
            active_manager = self.content_manager # 默认选择内容管理器
            
            query_embedding = self._generate_query_embedding(query_text)
            # 检查是否需要嵌入但生成失败
            if search_method in ["vector", "hybrid_rrf", "hybrid_weighted"] and query_embedding is None:
                self.logger.error(f"搜索方法 '{search_method}' 需要嵌入向量，但生成失败。")
                return []

            raw_results: Any = None
            try:
                # 确定单阶段搜索的输出字段
                single_stage_output_fields = [
                    f['name'] for f in active_manager.schema_fields_config 
                    if f['name'] not in [active_manager.embedding_field_name, active_manager.tokenized_text_field_name_for_bm25]
                ]
                # 确保唯一ID字段被包含
                if active_manager.unique_id_field_for_fusion not in single_stage_output_fields:
                     single_stage_output_fields.append(active_manager.unique_id_field_for_fusion)

                # 根据 search_method 调用 active_manager 的相应方法
                if search_method == "vector":
                    raw_results = active_manager.search_vector(query_embeddings=query_embedding, top_k=top_k, expr=filters_expr, output_fields=list(set(single_stage_output_fields)))
                elif search_method == "bm25":
                    raw_results = active_manager.search_bm25_only(query_text=query_text, top_k=top_k, expr=filters_expr, output_fields=list(set(single_stage_output_fields)))
                elif search_method in ["hybrid_rrf", "hybrid_weighted"]:
                    fusion_m = "rrf" if search_method == "hybrid_rrf" else "weighted"
                    raw_results = active_manager.hybrid_search(query_embeddings=query_embedding, query_text=query_text, top_k=top_k, fusion_method=fusion_m, vector_weight=vector_weight, bm25_weight=bm25_weight, expr=filters_expr, output_fields=list(set(single_stage_output_fields)))
                else: 
                    self.logger.error(f"未知的单阶段搜索方法: {search_method}")
                    return []
                
                # 格式化单阶段结果
                return self._format_single_stage_results(raw_results, search_method, active_manager.unique_id_field_for_fusion, active_manager == self.content_manager)
            except Exception as e: 
                self.logger.error(f"执行单阶段 '{search_method}' 搜索时发生错误: {e}", exc_info=True)
                return []

    def _format_single_stage_results(self, raw_results: Any, search_method: str, unique_id_field: str, is_content_collection: bool) -> List[Dict]:
        """ 格式化单阶段搜索的结果。 """
        formatted_list: List[Dict[str, Any]] = []
        items_to_process: List[Union[Dict, Any]] = [] 
        
        # 根据不同搜索方法返回的原始结果类型进行处理
        if search_method == "vector": 
            # 假设 MilvusManager.search_vector 返回 [[Hit,...]]
            if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list): 
                items_to_process = raw_results[0] # 取第一个查询的结果列表
            # 兼容可能直接返回 [Hit,...] 的情况
            elif isinstance(raw_results, list) and len(raw_results) > 0 and hasattr(raw_results[0], 'entity'):
                 items_to_process = raw_results
        elif search_method in ["bm25", "hybrid_rrf", "hybrid_weighted"]: 
            # 假设这些方法返回 List[Dict]
            if isinstance(raw_results, list): 
                items_to_process = raw_results
        
        if not items_to_process: 
            self.logger.info(f"单阶段搜索 '{search_method}' 没有可格式化的结果。")
            return []

        for item_or_hit in items_to_process:
            formatted_item: Dict[str, Any] = {}
            entity_data: Optional[Dict[str, Any]] = None
            score = 0.0

            # 统一处理 Hit 对象和字典
            if hasattr(item_or_hit, 'entity') and hasattr(item_or_hit, 'distance'): # 假设是 Hit 对象
                try:
                    hit_obj = item_or_hit 
                    entity_data = hit_obj.entity # entity 应该已经是字典
                    score = 1.0 - float(hit_obj.distance) # 假设 COSINE 相似度
                except Exception as e_hit:
                    self.logger.warning(f"处理疑似 Hit 对象时出错: {e_hit}. 对象: {str(item_or_hit)[:100]}")
                    continue # 跳过此项
            elif isinstance(item_or_hit, dict): # 来自 hybrid_search 或 bm25_only
                entity_data = item_or_hit
                score = float(item_or_hit.get('score', 0.0)) # 假设字典中有 score
            else: 
                self.logger.warning(f"单阶段格式化：遇到未知类型结果项 {type(item_or_hit)}")
                continue
            
            if entity_data:
                
                # 使用传入的 unique_id_field (例如 'chunk_id' 或 'question_unique_id')
                formatted_item['id'] = entity_data.get(unique_id_field) 
                formatted_item['score'] = score
                
                # 根据是内容集合还是问题集合，填充特定字段
                if is_content_collection:
                    formatted_item['content'] = entity_data.get(CONTENT_TEXT_FIELD, 'N/A') # 使用常量
                    formatted_item['doc_title'] = entity_data.get('doc_title', 'N/A')
                    # 确保 chunk_id (即 CONTENT_UNIQUE_ID_FIELD) 被包含
                    if CONTENT_UNIQUE_ID_FIELD not in formatted_item and CONTENT_UNIQUE_ID_FIELD in entity_data:
                        formatted_item[CONTENT_UNIQUE_ID_FIELD] = entity_data[CONTENT_UNIQUE_ID_FIELD]
                else: # is_questions_collection
                    formatted_item['question'] = entity_data.get(QUESTIONS_TEXT_FIELD, 'N/A') # 使用常量
                    formatted_item['chunk_id'] = entity_data.get("chunk_id", 'N/A') # 直接使用字段名
                    # 确保 QUESTIONS_UNIQUE_ID_FIELD 被包含
                    if QUESTIONS_UNIQUE_ID_FIELD not in formatted_item and QUESTIONS_UNIQUE_ID_FIELD in entity_data:
                        formatted_item[QUESTIONS_UNIQUE_ID_FIELD] = entity_data[QUESTIONS_UNIQUE_ID_FIELD]

                # 填充所有其他原始字段 (避免覆盖 id 和 score)
                for k, v in entity_data.items():
                    if k not in formatted_item: 
                        formatted_item[k] = v
                formatted_list.append(formatted_item)
        
        formatted_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        self.logger.info(f"单阶段 '{search_method}' 成功格式化 {len(formatted_list)} 个结果。")
        return formatted_list