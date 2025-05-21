# -*- coding: utf-8 -*-

import os
import numpy as np
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility, MilvusException
)
from rank_bm25 import BM25Okapi
import jieba 
import gc
import time
import logging
try:
    from Utils.Text_clean import TextPreprocessor
    TEXT_PREPROCESSOR_AVAILABLE = True
except ImportError:
    class TextPreprocessor:
        def __init__(self, stopwords_path="/home/cdipd-admin/RAG_LeaderSpeak/Utils/cn_stopWord.txt"): # 停用词路径示例
            if stopwords_path and not TEXT_PREPROCESSOR_AVAILABLE:
                 print(f"启动警告: TextPreprocessor 不可用，无法加载停用词: {stopwords_path}")
            self.stopwords = set()
            if stopwords_path:
                try:
                    with open(stopwords_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            self.stopwords.add(line.strip())
                    if self.stopwords:
                        print(f"占位符 TextPreprocessor: 已从 {stopwords_path} 加载 {len(self.stopwords)} 个停用词。")
                except Exception as e:
                    print(f"占位符 TextPreprocessor: 从 {stopwords_path} 加载停用词失败: {e}")

        def preprocess_text(self, text):
            if text is None: return ""
            return str(text).strip().lower() # 示例：转小写并去除首尾空格

        def remove_stopwords(self, word_list):
            if not self.stopwords:
                return word_list
            return [word for word in word_list if word not in self.stopwords]

    TEXT_PREPROCESSOR_AVAILABLE = False
    logging.warning("未能找到或导入 TextPreprocessor。将仅使用 jieba 分词和基础清理。")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 日志格式保持不变，但消息内容会变中文
    datefmt='%Y-%m-%d %H:%M:%S'
)

module_logger = logging.getLogger(__name__) # 模块级日志记录器



class MilvusManager:
    """
    通用的 Milvus 向量数据库管理器。
    """
    def __init__(self, 
                 dimension, 
                 collection_name,
                 schema_fields_config, 
                 schema_description,
                 embedding_field_name, 
                 text_field_name_for_bm25, 
                 tokenized_text_field_name_for_bm25,
                 unique_id_field_for_fusion,
                 host="localhost", port="19530",
                 stopwords_path="/home/cdipd-admin/RAG_LeaderSpeak/Utils/cn_stopWord.txt", # 默认停用词路径
                 user_dict_path="/home/cdipd-admin/RAG_LeaderSpeak/Utils/规划词库V6.txt",   # 默认用户词典路径
                 log_level=logging.INFO):
        
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("必须提供有效的正整数向量维度。")
        if not collection_name:
            raise ValueError("必须提供集合名称。")
       
        self.dimension = dimension
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.connection_alias = f"conn_{self.collection_name}_{os.getpid()}" # 连接别名

        self.schema_fields_config = schema_fields_config
        self.schema_description = schema_description
        self.embedding_field_name = embedding_field_name
        self.text_field_name_for_bm25 = text_field_name_for_bm25
        self.tokenized_text_field_name_for_bm25 = tokenized_text_field_name_for_bm25
        self.unique_id_field_for_fusion = unique_id_field_for_fusion

        self.schema_fields = []
        for field_conf in self.schema_fields_config:
            params = field_conf.copy()
            if params['name'] == self.embedding_field_name:
                params['dim'] = self.dimension
            self.schema_fields.append(FieldSchema(**params))
        
        self.schema_field_names_ordered = [field.name for field in self.schema_fields]

        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.collection_name}")
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            self.logger.propagate = True # 确保日志消息传递到根日志记录器的处理程序
        self.logger.info(f"初始化 MilvusManager: 集合 '{collection_name}' (维度={dimension})")
        self.logger.info(f"用于 BM25 的文本字段: '{self.text_field_name_for_bm25}', 分词后字段: '{self.tokenized_text_field_name_for_bm25}'")
        self.logger.info(f"嵌入向量字段: '{self.embedding_field_name}'")
        self.logger.info(f"用于融合的唯一ID字段: '{self.unique_id_field_for_fusion}'")

        # 加载用户自定义词典
        if user_dict_path:
            try:
                if os.path.exists(user_dict_path):
                    jieba.load_userdict(user_dict_path)
                    self.logger.info(f"用户自定义词典已从 {user_dict_path} 加载。")
                else:
                    self.logger.warning(f"用户自定义词典文件未找到: {user_dict_path}。将使用默认jieba分词。")
            except Exception as e:
                self.logger.warning(f"加载用户自定义词典 {user_dict_path} 失败: {e}。将使用默认jieba分词。", exc_info=False)
        else:
            self.logger.info("未提供用户自定义词典路径，使用默认jieba分词。")

        # 初始化文本预处理器
        if TEXT_PREPROCESSOR_AVAILABLE and stopwords_path:
            try:
                self.text_preprocessor = TextPreprocessor(stopwords_path)
                self.logger.info(f"TextPreprocessor 已使用停用词表 {stopwords_path} 初始化。")
            except Exception as e:
                self.logger.warning(f"初始化 TextPreprocessor 失败 (路径: {stopwords_path}): {e}。将使用基础处理器。", exc_info=False)
                self.text_preprocessor = TextPreprocessor() # 使用占位符
        else:
            self.text_preprocessor = TextPreprocessor(stopwords_path=stopwords_path if not TEXT_PREPROCESSOR_AVAILABLE else None)
            if not TEXT_PREPROCESSOR_AVAILABLE:
                 self.logger.warning("TextPreprocessor 不可用。将使用基础文本处理和jieba分词。")
            elif not stopwords_path:
                 self.logger.info("未提供停用词路径。将使用基础文本处理和jieba分词。")

        self._connect()
        self._check_and_load_collection()

    def _connect(self):
        self.logger.info(f"尝试连接到 Milvus: {self.host}:{self.port} (别名: {self.connection_alias})")
        try:
            if self.connection_alias in connections.list_connections():
                self.logger.debug(f"发现已存在的连接别名 {self.connection_alias}，将断开并重连。")
                connections.disconnect(self.connection_alias)
            connections.connect(self.connection_alias, host=self.host, port=self.port, timeout=10) # 添加超时
            self.logger.info(f"成功连接到 Milvus (别名: {self.connection_alias})")
        except MilvusException as e:
            self.logger.error(f"连接 Milvus 失败: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"连接 Milvus 时发生意外错误: {e}", exc_info=True)
            raise

    def _check_and_load_collection(self):
        self.logger.debug(f"检查集合 '{self.collection_name}' 是否存在...")
        try:
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                self.logger.info(f"找到已存在的集合: {self.collection_name}。正在加载 Collection 对象...")
                current_schema_fields = []
                for field_conf in self.schema_fields_config:
                    params = field_conf.copy()
                    if params['name'] == self.embedding_field_name:
                        params['dim'] = self.dimension
                    current_schema_fields.append(FieldSchema(**params))
                schema = CollectionSchema(current_schema_fields, self.schema_description)
                self.collection = Collection(self.collection_name, schema=schema, using=self.connection_alias)
                self.logger.info(f"集合对象 '{self.collection_name}' 已加载。")
            else:
                self.logger.info(f"集合 '{self.collection_name}' 不存在。请调用 create_collection() 创建。")
                self.collection = None
        except MilvusException as e:
            self.logger.error(f"检查或加载集合 '{self.collection_name}' 时出错: {e}", exc_info=True)
            self.collection = None
        except Exception as e:
             self.logger.error(f"检查或加载集合时发生意外错误: {e}", exc_info=True)
             self.collection = None

    def is_loaded(self):
        if self.collection is None:
            self.logger.debug("集合对象未初始化，无法检查加载状态。")
            return False
        self.logger.debug(f"正在检查集合 '{self.collection_name}' 的内存加载状态...")
        try:
            load_state = utility.load_state(self.collection_name, using=self.connection_alias)
            is_loaded_flag = str(load_state).endswith("Loaded") 
            self.logger.debug(f"集合加载状态: {load_state} -> {'已加载' if is_loaded_flag else '未加载'}")
            return is_loaded_flag
        except MilvusException as e:
             if "does not exist" in str(e).lower() or "not found" in str(e).lower(): # 更通用的错误检查
                  self.logger.warning(f"集合 {self.collection_name} 不存在，无法检查加载状态。")
             else:
                  self.logger.error(f"检查集合加载状态时出错: {e}", exc_info=False)
             return False
        except Exception as e:
            self.logger.error(f"检查集合加载状态时发生未知错误: {e}", exc_info=True)
            return False

    def create_collection(self, drop_existing=False, scalar_index_fields=None):
        collection_exists = utility.has_collection(self.collection_name, using=self.connection_alias)
        self.logger.info(f"尝试创建集合 '{self.collection_name}' (已存在: {collection_exists}, drop_existing={drop_existing})")

        if collection_exists:
            if drop_existing:
                self.logger.warning(f"集合 {self.collection_name} 已存在且 drop_existing=True。将删除并重建。")
                try:
                     self.drop_collection(confirm_prompt=False) # 内部调用，不提示确认
                except Exception as e:
                     self.logger.error(f"删除现有集合 '{self.collection_name}' 失败。无法继续创建: {e}", exc_info=True)
                     return
            else:
                self.logger.info(f"集合 {self.collection_name} 已存在且 drop_existing=False。跳过创建。")
                if self.collection is None: self._check_and_load_collection()
                return

        schema = CollectionSchema(self.schema_fields, self.schema_description, enable_dynamic_field=False)
        try:
            self.logger.info(f"执行创建集合: {self.collection_name}...")
            self.collection = Collection(self.collection_name, schema, using=self.connection_alias)
            self.logger.info(f"成功创建集合: {self.collection_name}")

            default_scalar_fields_to_index = ["source_file", self.unique_id_field_for_fusion]
            if scalar_index_fields is None:
                scalar_index_fields_to_create = default_scalar_fields_to_index
            else:
                scalar_index_fields_to_create = list(set(default_scalar_fields_to_index + scalar_index_fields))
            
            field_names_in_schema = [f.name for f in self.schema_fields]
            for field_name in scalar_index_fields_to_create:
                if field_name in field_names_in_schema and field_name != self.embedding_field_name:
                    field_schema = next((f for f in self.schema_fields if f.name == field_name), None)
                    if field_schema and field_schema.dtype in [DataType.VARCHAR, DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.BOOL, DataType.FLOAT, DataType.DOUBLE]:
                        try:
                            self.logger.info(f"为字段 '{field_name}' 创建标量索引...")
                            self.collection.create_index(field_name=field_name, index_name=f"idx_{field_name}")
                            self.logger.info(f"字段 '{field_name}' 的标量索引已创建或已存在。")
                        except MilvusException as idx_e:
                            if "already exist" in str(idx_e): self.logger.info(f"字段 '{field_name}' 的标量索引已存在。")
                            else: self.logger.warning(f"为字段 '{field_name}' 创建标量索引时出错: {idx_e}", exc_info=False)
                        except Exception as idx_e: self.logger.warning(f"为字段 '{field_name}' 创建标量索引时发生意外错误: {idx_e}", exc_info=True)
                    else: self.logger.warning(f"字段 '{field_name}' 类型不支持标量索引或未找到，跳过创建。")
                elif field_name != self.embedding_field_name : self.logger.warning(f"字段 '{field_name}' 未在Schema中找到，无法创建标量索引。")

        except MilvusException as e:
            self.logger.error(f"创建集合 '{self.collection_name}' 时出错: {e}", exc_info=True)
            self.collection = None; raise
        except Exception as e:
             self.logger.error(f"创建集合时发生意外错误: {e}", exc_info=True)
             self.collection = None; raise

    def create_vector_index(self, index_type="HNSW", metric_type="COSINE", params=None):
        if self.collection is None:
             self._check_and_load_collection()
             if self.collection is None:
                 self.logger.error(f"集合 {self.collection_name} 未创建或加载失败。无法创建向量索引。")
                 raise ValueError(f"集合 {self.collection_name} 不可用。")

        self.logger.info(f"准备为字段 '{self.embedding_field_name}' 创建向量索引 (类型: {index_type}, 度量: {metric_type})")
        try:
            existing_indexes = self.collection.indexes
            for index in existing_indexes:
                if index.field_name == self.embedding_field_name:
                    self.logger.info(f"字段 '{self.embedding_field_name}' 已存在向量索引: {index.index_name}。跳过创建。")
                    return
        except MilvusException as e: self.logger.warning(f"检查现有索引时出错: {e}", exc_info=False)

        if params is None: # 根据索引类型设置默认参数
            if index_type == "HNSW": params = {"M": 16, "efConstruction": 256}
            elif index_type == "IVF_FLAT": params = {"nlist": 1024} # IVF_FLAT 需要 nlist
            # 为其他索引类型添加更多默认参数
            else: params = {}
        index_params = {"metric_type": metric_type, "index_type": index_type, "params": params}
        index_name = f"idx_{self.embedding_field_name}_{index_type.lower()}"

        self.logger.info(f"提交创建索引任务: 名称='{index_name}', 参数={index_params}")
        try:
            self.collection.create_index(self.embedding_field_name, index_params, index_name=index_name, timeout=3600) # 增加超时
            self.logger.info(f"向量索引创建任务 '{index_name}' 已提交。")
        except MilvusException as e: self.logger.error(f"创建向量索引 '{index_name}' 失败: {e}", exc_info=True); raise
        except Exception as e: self.logger.error(f"创建向量索引时发生意外错误: {e}", exc_info=True); raise

    def add_data(self, embeddings, data_list):
        """向 Milvus 集合添加数据。"""
        if self.collection is None:
             self._check_and_load_collection()
             if self.collection is None:
                 self.logger.error(f"集合 {self.collection_name} 未创建或加载失败。无法添加数据。")
                 raise ValueError(f"集合 {self.collection_name} 不可用。")

        if not data_list: self.logger.warning("输入的数据列表 data_list 为空。不插入任何数据。"); return None

        try: embeddings_np = np.array(embeddings, dtype='float32')
        except ValueError as e: self.logger.error(f"无法将嵌入向量转换为 NumPy 数组: {e}", exc_info=True); raise ValueError("嵌入向量格式无效。") from e

        if len(embeddings_np) != len(data_list):
            self.logger.error(f"嵌入向量数量 ({len(embeddings_np)}) 与数据项数量 ({len(data_list)}) 不匹配。")
            raise ValueError("嵌入向量数量必须与数据项数量匹配。")
        if embeddings_np.ndim != 2 or embeddings_np.shape[1] != self.dimension:
            self.logger.error(f"嵌入向量维度 ({embeddings_np.shape}) 与集合维度 ({self.dimension}) 不匹配或格式错误。")
            raise ValueError("嵌入向量维度错误。")

        # 根据 Schema 字段名初始化 field_data
        field_data = {name: [] for name in self.schema_field_names_ordered}
        # 如果主键是 auto_id，则从待插入数据中移除
        pk_field = self.collection.schema.primary_field
        if pk_field.auto_id and pk_field.name in field_data: del field_data[pk_field.name]
        
        self.logger.info(f"准备处理 {len(data_list)} 条数据项用于插入...")
        processed_count = 0
        for i, item_dict in enumerate(data_list):
            # 获取用于 BM25 的原始文本
            raw_text_for_bm25 = item_dict.get(self.text_field_name_for_bm25, "")
            if not raw_text_for_bm25: # 确保 BM25 的主要文本字段存在
                self.logger.warning(f"第 {i+1} 项数据缺少必需的文本字段 '{self.text_field_name_for_bm25}'。跳过此项。数据: {str(item_dict)[:100]}...")
                continue
            try:
                clean_text = self.text_preprocessor.preprocess_text(raw_text_for_bm25)
                tokens = list(jieba.cut(clean_text)) # jieba.cut 会受 load_userdict 影响
                tokens = self.text_preprocessor.remove_stopwords(tokens)
                tokenized_text_str = " ".join(tokens)
            except Exception as e:
                 self.logger.warning(f"为第 {i+1} 项数据进行文本分词时出错 (字段: {self.text_field_name_for_bm25}): {e}。将使用空的分词结果。", exc_info=False)
                 tokenized_text_str = ""

            # 填充每个字段的数据
            for field_name in field_data.keys():
                if field_name == self.embedding_field_name: field_data[field_name].append(embeddings_np[processed_count].tolist()) # 使用 processed_count 索引嵌入
                elif field_name == self.tokenized_text_field_name_for_bm25: field_data[field_name].append(tokenized_text_str)
                else:
                    # 从 item_dict 获取值，如果不存在则使用 Schema 中定义的默认值
                    field_schema_conf = next((f for f in self.schema_fields_config if f['name'] == field_name), None)
                    default_val = field_schema_conf.get('default_value', None) if field_schema_conf else None
                    val = item_dict.get(field_name, default_val)
                    # 类型转换和默认值处理
                    if field_schema_conf and field_schema_conf['dtype'] == DataType.INT64 and not isinstance(val, int) and val is not None:
                        try: val = int(val)
                        except (ValueError, TypeError): self.logger.warning(f"无法将值 '{val}' 转换为整数 (字段: '{field_name}')。使用默认值 {default_val}。"); val = default_val if default_val is not None else 0
                    elif val is None and default_val is not None: val = default_val # 如果值为None，显式使用默认值
                    elif val is None and field_schema_conf and field_schema_conf['dtype'] in [DataType.VARCHAR]: val = "" # VARCHAR 的 None 值转为空字符串
                    field_data[field_name].append(val)
            processed_count += 1
        
        if processed_count == 0: self.logger.warning("处理后没有有效的数据项可供插入。"); return None
        self.logger.info(f"成功处理 {processed_count} 项数据。原始数据项数量: {len(data_list)}。")

        # 确保插入的实体列表顺序与 Schema 一致 (不包括 auto_id 主键)
        entities_to_insert = []
        current_field_names_for_insert = [] # 用于日志记录实际插入的字段
        for name in self.schema_field_names_ordered:
            pk_field_schema = self.collection.schema.primary_field
            if pk_field_schema.auto_id and name == pk_field_schema.name: continue # 跳过 auto_id 主键
            if name in field_data: entities_to_insert.append(field_data[name]); current_field_names_for_insert.append(name)
            else: # 此情况理论上不应发生，因为 field_data 是基于 schema_field_names_ordered 初始化的
                 self.logger.error(f"逻辑错误: 字段 '{name}' 未在准备好的 field_data 中找到。"); field_schema_conf = next((f for f in self.schema_fields_config if f['name'] == name), None); default_val_list = [field_schema_conf.get('default_value', None) if field_schema_conf else None] * processed_count; entities_to_insert.append(default_val_list); current_field_names_for_insert.append(name)

        self.logger.info(f"准备向 Milvus 插入 {processed_count} 条有效数据。插入字段: {current_field_names_for_insert}")
        try:
            insert_result = self.collection.insert(entities_to_insert)
            self.logger.info("数据插入请求已发送。正在执行 flush 操作..."); start_time = time.time(); self.collection.flush(); flush_time = time.time() - start_time
            self.logger.info(f"Flush 操作完成 (耗时 {flush_time:.2f} 秒)。成功插入 {insert_result.insert_count} 条数据。")
            return insert_result
        except MilvusException as e: self.logger.error(f"向 Milvus 插入数据时出错: {e}", exc_info=True); raise
        except Exception as e: self.logger.error(f"插入数据时发生意外错误: {e}", exc_info=True); raise

    def search_vector(self, query_embeddings, top_k=10, expr=None, output_fields=None, metric_type="COSINE"):
        """执行基于嵌入向量的相似性搜索。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error("集合未初始化或加载失败，无法执行向量搜索。"); return []
        if not self.is_loaded():
            self.logger.warning("集合数据未加载到内存，搜索性能可能较低。尝试加载集合...")
            try: self.load_collection()
            except Exception as load_e: self.logger.warning(f"尝试加载集合失败: {load_e}。继续搜索...", exc_info=False)
        if output_fields is None: # 默认输出所有非嵌入向量字段
            output_fields = [f.name for f in self.schema_fields if f.name != self.embedding_field_name]
        self.logger.debug(f"向量搜索输出字段: {output_fields}")
        try:
            query_embeddings_np = np.array(query_embeddings, dtype='float32')
            if query_embeddings_np.ndim == 1: query_embeddings_np = query_embeddings_np.reshape(1, -1) # 单个查询向量 reshape
            elif query_embeddings_np.ndim != 2: raise ValueError("查询嵌入向量必须是一维或二维数组。")
            if query_embeddings_np.shape[1] != self.dimension: raise ValueError(f"查询嵌入向量维度 ({query_embeddings_np.shape[1]}) 与集合维度不匹配。")
        except ValueError as e: self.logger.error(f"查询嵌入向量处理失败: {e}", exc_info=True); raise
        search_params = {"metric_type": metric_type, "params": {"ef": max(top_k * 2, 128)}} # ef 参数可调优
        self.logger.info(f"执行向量搜索: top_k={top_k}, expr='{expr or '无表达式'}'")
        self.logger.debug(f"搜索参数: {search_params}")
        try:
            start_time = time.time()
            search_results = self.collection.search(data=query_embeddings_np.tolist(), anns_field=self.embedding_field_name, param=search_params, limit=top_k, expr=expr, output_fields=output_fields, timeout=30) # 添加搜索超时
            search_time = time.time() - start_time; result_count = len(search_results[0]) if search_results and len(search_results) > 0 else 0
            self.logger.info(f"向量搜索完成 (耗时 {search_time:.3f} 秒)。为第一个查询找到 {result_count} 个结果。")
            return search_results
        except MilvusException as e: self.logger.error(f"向量搜索时出错: {e}", exc_info=True); return []
        except Exception as e: self.logger.error(f"向量搜索时发生意外错误: {e}", exc_info=True); return []

    def hybrid_search(self, query_embeddings, query_text, top_k=10, expr=None, output_fields=None,
                         fusion_method="rrf", rrf_k=60, bm25_weight=0.3, vector_weight=0.7,
                         bm25_candidate_factor=20, vector_search_candidate_factor=5):
        """执行混合搜索：结合向量搜索和 BM25 关键词搜索。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error("集合未初始化或加载失败，无法执行混合搜索。"); raise ValueError("集合不可用。")
        if not self.is_loaded():
            self.logger.warning("集合数据未加载到内存，混合搜索性能可能受影响。尝试加载...")
            try: self.load_collection()
            except Exception as load_e: self.logger.warning(f"加载集合失败: {load_e}。继续执行...", exc_info=False)
        if output_fields is None: # 默认输出所有非嵌入和非分词文本字段
            output_fields = [f.name for f in self.schema_fields if f.name != self.embedding_field_name and f.name != self.tokenized_text_field_name_for_bm25]
        # 确保唯一ID字段和主键字段被获取，用于融合和结果处理
        if self.unique_id_field_for_fusion not in output_fields: output_fields.append(self.unique_id_field_for_fusion)
        if self.collection.schema.primary_field.name not in output_fields: output_fields.append(self.collection.schema.primary_field.name)
        self.logger.debug(f"混合搜索输出字段: {output_fields}")

        # --- 1. 向量搜索 ---
        vector_search_top_k = top_k * vector_search_candidate_factor # 获取更多候选结果
        self.logger.info(f"--- 混合搜索阶段 1: 向量搜索 (Top {vector_search_top_k}) ---")
        vector_search_output_fields = list(set(output_fields + [self.collection.schema.primary_field.name, self.unique_id_field_for_fusion])) # 确保向量搜索获取必要字段
        vector_search_results_raw = self.search_vector(query_embeddings, top_k=vector_search_top_k, expr=expr, output_fields=vector_search_output_fields)
        flat_vector_hits = [hit for hits_per_query in vector_search_results_raw for hit in hits_per_query] if vector_search_results_raw else [] # 扁平化结果列表
        self.logger.info(f"向量搜索召回 {len(flat_vector_hits)} 个候选结果。")

        # --- 2. BM25 关键词搜索 ---
        keyword_results_ranked = []
        if query_text and query_text.strip(): # 仅当查询文本非空时执行
            self.logger.info(f"--- 混合搜索阶段 2: BM25 关键词搜索 (查询: '{query_text[:50]}...') ---")
            # BM25 需要主键, 唯一ID字段, 分词文本字段, 以及用户指定的输出字段
            bm25_fetch_fields = list(set([self.collection.schema.primary_field.name, self.unique_id_field_for_fusion, self.tokenized_text_field_name_for_bm25] + output_fields))
            bm25_candidate_limit = max(top_k * bm25_candidate_factor, 500) # 为 BM25 获取足够候选文档
            self.logger.info(f"正在查询用于 BM25 的文档 (最多 {bm25_candidate_limit} 条)，匹配表达式: '{expr or '无表达式'}'...")
            try:
                query_start_time = time.time()
                all_docs_for_bm25 = self.collection.query(expr=expr if expr else "", output_fields=bm25_fetch_fields, limit=bm25_candidate_limit, consistency_level="Strong") # "Strong"确保最新数据
                query_time = time.time() - query_start_time
                self.logger.info(f"获取了 {len(all_docs_for_bm25)} 个 BM25 候选文档 (耗时 {query_time:.3f} 秒)。")
                if all_docs_for_bm25:
                    bm25_start_time = time.time(); bm25_top_k_internal = top_k * vector_search_candidate_factor # 内部BM25排序数量
                    keyword_results_ranked = self._bm25_rank_docs(query_text, all_docs_for_bm25, top_k=bm25_top_k_internal)
                    bm25_time = time.time() - bm25_start_time
                    self.logger.info(f"BM25 排序完成 (耗时 {bm25_time:.3f} 秒)。找到 {len(keyword_results_ranked)} 个相关结果。")
                else: self.logger.info("未找到可用于 BM25 搜索的候选文档。")
            except MilvusException as e: self.logger.error(f"查询 BM25 候选文档时出错: {e}", exc_info=True)
            except Exception as e: self.logger.error(f"获取或处理 BM25 候选文档时发生意外错误: {e}", exc_info=True)
        else: self.logger.info("未提供查询文本，跳过 BM25 搜索阶段。")

        # --- 处理边界情况 (一种或两种搜索无结果) ---
        if not flat_vector_hits and not keyword_results_ranked: self.logger.info("向量搜索和 BM25 均无结果。"); return []
        if not flat_vector_hits: self.logger.info("仅 BM25 有结果，返回 BM25 排序结果。"); return [ {k: v for k, v in doc.items() if k in output_fields} for doc in keyword_results_ranked[:top_k] ] # BM25结果已是字典
        if not keyword_results_ranked: self.logger.info("仅向量搜索有结果，返回向量排序结果。"); sorted_hits = sorted(flat_vector_hits, key=lambda hit: hit.distance); return [ {k:v for k,v in hit.entity.to_dict()['entity'].items() if k in output_fields} for hit in sorted_hits[:top_k] ] # 向量结果是Hit对象

        # --- 3. 结果融合 ---
        self.logger.info(f"--- 混合搜索阶段 3: 结果融合 ({fusion_method}) ---")
        fusion_start_time = time.time(); fused_items = []
        try:
            if fusion_method == "rrf": fused_items = self._rrf_fusion(flat_vector_hits, keyword_results_ranked, top_k, rrf_k)
            elif fusion_method == "weighted":
                if not (0 <= vector_weight <= 1 and 0 <= bm25_weight <= 1 and (vector_weight > 0 or bm25_weight > 0)): self.logger.warning(f"融合权重无效 (向量: {vector_weight}, BM25: {bm25_weight})。将使用默认值 0.7, 0.3"); vector_weight, bm25_weight = 0.7, 0.3
                fused_items = self._weighted_fusion(flat_vector_hits, keyword_results_ranked, top_k, [vector_weight, bm25_weight])
            else: self.logger.error(f"不支持的融合方法: {fusion_method}。将使用 RRF 作为默认方法。"); fused_items = self._rrf_fusion(flat_vector_hits, keyword_results_ranked, top_k, rrf_k)
            fusion_time = time.time() - fusion_start_time
            self.logger.info(f"融合完成 (耗时 {fusion_time:.4f} 秒)。融合后得到 {len(fused_items)} 个候选结果。")
        except Exception as e: # 融合失败时的回退策略
             self.logger.error(f"结果融合时出错: {e}", exc_info=True); self.logger.warning("融合失败。将返回排序后的向量搜索结果作为备选。"); sorted_hits = sorted(flat_vector_hits, key=lambda hit: hit.distance); return [ {k:v for k,v in hit.entity.to_dict()['entity'].items() if k in output_fields} for hit in sorted_hits[:top_k] ]

        # --- 4. 格式化输出 ---
        self.logger.info("正在格式化并去重最终结果..."); final_results_list = []; seen_unique_ids = set() # 使用配置的唯一ID字段去重
        for item_wrapper in fused_items: # 融合项是 {'score': ..., 'item': Hit对象或字典}
            item_data = item_wrapper['item']; result_dict = {}; item_unique_id = None
            try:
                if isinstance(item_data, dict): result_dict = item_data; item_unique_id = result_dict.get(self.unique_id_field_for_fusion) # 来自BM25或已处理
                elif hasattr(item_data, 'entity') and hasattr(item_data.entity, 'to_dict'): result_dict = item_data.entity.to_dict()['entity']; item_unique_id = result_dict.get(self.unique_id_field_for_fusion) # Milvus Hit 对象
                else: self.logger.warning(f"融合结果中存在未知类型的项: {type(item_data)}。跳过此项。"); continue
                if item_unique_id and item_unique_id not in seen_unique_ids: # 基于唯一ID去重
                    formatted_dict = {k: v for k, v in result_dict.items() if k in output_fields} # 仅保留请求的输出字段
                    final_results_list.append(formatted_dict); seen_unique_ids.add(item_unique_id)
                elif not item_unique_id: self.logger.warning(f"结果项缺少唯一ID字段 '{self.unique_id_field_for_fusion}'。数据项: {str(result_dict)[:100]}")
                if len(final_results_list) >= top_k: break # 已达到所需数量
            except Exception as format_e: self.logger.warning(f"格式化融合结果项时出错: {format_e}。数据项: {str(item_data)[:100]}", exc_info=False); continue
        self.logger.info(f"最终返回 {len(final_results_list)} 条去重后的结果。"); return final_results_list

    def search_bm25_only(self, query_text, top_k=10, expr=None, output_fields=None, bm25_candidate_factor=30):
        """仅执行 BM25 关键词搜索。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error("集合未初始化或加载失败，无法执行 BM25 搜索。"); raise ValueError("集合不可用。")
        if not query_text or not query_text.strip(): self.logger.warning("查询文本为空，无法执行 BM25 搜索。"); return []
        self.logger.info(f"--- 开始纯 BM25 搜索 (查询: '{query_text[:50]}...') ---")
        if output_fields is None: output_fields = [f.name for f in self.schema_fields if f.name != self.embedding_field_name and f.name != self.tokenized_text_field_name_for_bm25]
        bm25_fetch_fields = list(set([self.collection.schema.primary_field.name, self.unique_id_field_for_fusion, self.tokenized_text_field_name_for_bm25] + output_fields))
        bm25_candidate_limit = max(top_k * bm25_candidate_factor, 200) # 为纯 BM25 获取更多候选以提高排序准确性
        self.logger.info(f"正在查询用于 BM25 计算的文档 (最多 {bm25_candidate_limit} 条)，匹配表达式: '{expr or '无表达式'}'...")
        try:
            query_start_time = time.time()
            all_docs_for_bm25 = self.collection.query(expr=expr if expr else "", output_fields=bm25_fetch_fields, limit=bm25_candidate_limit, consistency_level="Strong")
            query_time = time.time() - query_start_time
            self.logger.info(f"获取了 {len(all_docs_for_bm25)} 个 BM25 候选文档 (耗时 {query_time:.3f} 秒)。")
            if not all_docs_for_bm25: self.logger.info("未找到可用于 BM25 搜索的候选文档。"); return []
            bm25_start_time = time.time()
            ranked_keyword_results = self._bm25_rank_docs(query_text, all_docs_for_bm25, top_k=top_k) # 使用内部 BM25 排序方法
            bm25_time = time.time() - bm25_start_time
            self.logger.info(f"BM25 排序完成 (耗时 {bm25_time:.3f} 秒)。找到 {len(ranked_keyword_results)} 个相关结果。")
            final_results = [ {k: v for k, v in doc.items() if k in output_fields} for doc in ranked_keyword_results ] # 格式化输出
            return final_results
        except MilvusException as e: self.logger.error(f"查询 BM25 候选文档时出错: {e}", exc_info=True); return []
        except Exception as e: self.logger.error(f"执行纯 BM25 搜索时发生意外错误: {e}", exc_info=True); return []

    def _bm25_rank_docs(self, query_text, documents, top_k=20):
        """内部方法：对 Milvus 文档列表 (字典格式) 使用 BM25 进行排序。"""
        if not documents: return []
        self.logger.debug(f"开始对 {len(documents)} 个文档进行 BM25 排序。查询: '{query_text[:50]}...'")
        try:
            clean_query = self.text_preprocessor.preprocess_text(query_text)
            query_tokens = list(jieba.cut(clean_query)) # jieba.cut 受 load_userdict 影响
            query_tokens = self.text_preprocessor.remove_stopwords(query_tokens)
            if not query_tokens: self.logger.warning("查询文本在分词/去停用词后为空，无法进行 BM25 匹配。"); return []
        except Exception as e: self.logger.error(f"为 BM25 分词查询文本时出错: {e}", exc_info=True); return []
        corpus = []; doc_indices_map = [] # 语料库索引到原始文档列表索引的映射
        for i, doc in enumerate(documents): # doc 是从 Milvus 查询得到的字典
            tokenized_text_str = doc.get(self.tokenized_text_field_name_for_bm25, '') # 获取预分词的文本
            if tokenized_text_str and isinstance(tokenized_text_str, str):
                tokens = tokenized_text_str.split() # 按空格切分得到token列表
                if tokens: corpus.append(tokens); doc_indices_map.append(i)
        if not corpus: self.logger.debug("处理文档后，BM25 语料库为空。"); return []
        self.logger.debug(f"BM25 语料库已构建，包含 {len(corpus)} 个文档。")
        try: bm25 = BM25Okapi(corpus); scores = bm25.get_scores(query_tokens)
        except Exception as e: self.logger.error(f"计算 BM25 得分时出错: {e}", exc_info=True); return []
        doc_scores = [(documents[doc_indices_map[i]], scores[i]) for i in range(len(scores))] # 将原始文档与其BM25得分结合
        filtered_doc_scores = [(doc, score) for doc, score in doc_scores if score > 1e-9] # 过滤掉得分过低的结果
        sorted_docs_with_scores = sorted(filtered_doc_scores, key=lambda x: x[1], reverse=True) # 按得分降序排列
        self.logger.debug(f"BM25 排序后得到 {len(sorted_docs_with_scores)} 个得分大于0的结果。")
        return [doc for doc, score in sorted_docs_with_scores[:top_k]] # 返回排序后的文档字典列表

    def _rrf_fusion(self, vector_hits, keyword_results_ranked, top_k, k_param=60):
        """内部方法：RRF 结果融合。"""
        self.logger.debug(f"开始 RRF 融合 (k_param={k_param})。向量搜索结果: {len(vector_hits)} 条, 关键词搜索结果: {len(keyword_results_ranked)} 条")
        rrf_scores = {} # 键: 唯一ID, 值: {'score': float, 'item': Hit对象或字典}
        for rank, hit in enumerate(vector_hits): # 处理向量搜索结果 (Milvus Hit 对象)
            try:
                entity_dict = hit.entity.to_dict()['entity'] if hasattr(hit, 'entity') and hit.entity else {}
                item_id = entity_dict.get(self.unique_id_field_for_fusion)
                if item_id:
                    score = 1.0 / (k_param + rank + 1) # RRF 得分计算
                    if item_id not in rrf_scores: rrf_scores[item_id] = {'score': 0, 'item': hit} # 存储原始 Hit 对象
                    rrf_scores[item_id]['score'] += score
                else: self.logger.warning(f"向量搜索结果 (Milvus ID: {hit.id if hasattr(hit,'id') else 'N/A'}) 缺少唯一ID字段 '{self.unique_id_field_for_fusion}'。实体: {str(entity_dict)[:100]}")
            except Exception as e: self.logger.warning(f"处理 RRF 向量搜索结果时出错: {e}。数据项: {str(hit)[:100]}", exc_info=False)
        for rank, doc_dict in enumerate(keyword_results_ranked): # 处理关键词搜索结果 (字典)
            item_id = doc_dict.get(self.unique_id_field_for_fusion)
            if item_id:
                score = 1.0 / (k_param + rank + 1)
                if item_id not in rrf_scores: rrf_scores[item_id] = {'score': 0, 'item': doc_dict} # 存储字典
                rrf_scores[item_id]['score'] += score
            else: self.logger.warning(f"BM25 结果缺少唯一ID字段 '{self.unique_id_field_for_fusion}'。文档: {str(doc_dict)[:100]}")
        sorted_items_with_scores = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True) # 按RRF总分排序
        self.logger.debug(f"RRF 融合后得到 {len(sorted_items_with_scores)} 个候选结果。")
        return sorted_items_with_scores[:top_k] # 返回带得分和原始项的列表

    def _weighted_fusion(self, vector_hits, keyword_results_ranked, top_k, weights):
        """内部方法：加权结果融合。"""
        vector_weight, bm25_weight = weights
        self.logger.debug(f"开始加权融合 (权重 向量:{vector_weight}, BM25:{bm25_weight})。向量结果: {len(vector_hits)}, 关键词结果: {len(keyword_results_ranked)}")
        weighted_scores = {} # 键: 唯一ID, 值: {'score': float, 'item': Hit对象或字典}
        max_vec_hits_for_norm = len(vector_hits)
        if max_vec_hits_for_norm > 0: # 处理向量结果
            for rank, hit in enumerate(vector_hits):
                try:
                    entity_dict = hit.entity.to_dict()['entity'] if hasattr(hit, 'entity') and hit.entity else {}
                    item_id = entity_dict.get(self.unique_id_field_for_fusion)
                    if item_id:
                        # 假设COSINE距离 (0=相同, 2=相反)，转换为相似度 (0-1, 越高越好)
                        similarity_score = (2.0 - float(hit.distance)) / 2.0 if hasattr(hit, 'distance') else 0.0
                        similarity_score = max(0.0, min(1.0, similarity_score)) # 限制在 [0,1]
                        score_contribution = similarity_score * vector_weight
                        if item_id not in weighted_scores: weighted_scores[item_id] = {'score': 0, 'item': hit}
                        weighted_scores[item_id]['score'] += score_contribution
                    else: self.logger.warning(f"加权融合：向量结果 (Milvus ID: {hit.id if hasattr(hit,'id') else 'N/A'}) 缺少唯一ID字段 '{self.unique_id_field_for_fusion}'.")
                except Exception as e: self.logger.warning(f"处理加权融合的向量结果时出错: {e}。数据项: {str(hit)[:100]}", exc_info=False)
        max_keyword_rank_for_norm = len(keyword_results_ranked)
        if max_keyword_rank_for_norm > 0: # 处理关键词结果 (基于排名归一化)
            for rank, doc_dict in enumerate(keyword_results_ranked):
                item_id = doc_dict.get(self.unique_id_field_for_fusion)
                if item_id:
                    rank_score_norm = (max_keyword_rank_for_norm - rank) / max_keyword_rank_for_norm if max_keyword_rank_for_norm > 0 else 0 # 排名越高得分越高
                    score_contribution = rank_score_norm * bm25_weight
                    if item_id not in weighted_scores: weighted_scores[item_id] = {'score': 0, 'item': doc_dict}
                    weighted_scores[item_id]['score'] += score_contribution
                else: self.logger.warning(f"加权融合：BM25 结果缺少唯一ID字段 '{self.unique_id_field_for_fusion}'。文档: {str(doc_dict)[:100]}")
        sorted_items_with_scores = sorted(weighted_scores.values(), key=lambda x: x['score'], reverse=True)
        self.logger.debug(f"加权融合后得到 {len(sorted_items_with_scores)} 个候选结果。")
        return sorted_items_with_scores[:top_k]

    def load_collection(self):
        """将集合数据加载到内存。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error(f"集合 {self.collection_name} 未创建或加载失败，无法加载到内存。"); raise ValueError(f"集合 {self.collection_name} 不可用。")
        if not self.is_loaded():
            self.logger.info(f"正在加载集合 {self.collection_name} 的数据到内存...")
            try: start_time = time.time(); self.collection.load(); load_time = time.time() - start_time; self.logger.info(f"集合 {self.collection_name} 的数据已加载 (耗时 {load_time:.2f} 秒)。")
            except MilvusException as e: self.logger.error(f"加载集合数据时出错: {e}", exc_info=True); raise
            except Exception as e: self.logger.error(f"加载集合数据时发生意外错误: {e}", exc_info=True); raise
        else: self.logger.info(f"集合 {self.collection_name} 的数据已在内存中。")

    def release_collection(self):
        """从内存中释放集合数据。"""
        if self.collection is not None and self.is_loaded():
            self.logger.info(f"正在从内存释放集合 {self.collection_name} 的数据...")
            try: start_time = time.time(); self.collection.release(); release_time = time.time() - start_time; self.logger.info(f"集合 {self.collection_name} 的数据已释放 (耗时 {release_time:.2f} 秒)。")
            except MilvusException as e: self.logger.error(f"释放集合数据时出错: {e}", exc_info=True)
            except Exception as e: self.logger.error(f"释放集合数据时发生意外错误: {e}", exc_info=True)
        elif self.collection is None: self.logger.debug("集合对象未初始化，无需释放。")
        else: self.logger.debug(f"集合 {self.collection_name} 的数据未加载，无需释放。")

    def drop_collection(self, confirm_prompt=True):
        """删除整个集合 (危险操作！)。"""
        if utility.has_collection(self.collection_name, using=self.connection_alias):
            self.logger.warning(f"!!! 即将永久删除集合 {self.collection_name} !!!"); proceed = False
            if confirm_prompt: confirm = input(f"!!! 确认删除吗? 此操作不可逆！请输入 'yes' 确认删除集合 '{self.collection_name}': "); proceed = (confirm.lower() == 'yes')
            else: proceed = True # 无需提示，直接执行
            if proceed:
                try: self.logger.warning(f"正在执行删除集合 {self.collection_name}..."); utility.drop_collection(self.collection_name, using=self.connection_alias); self.collection = None; self.logger.info(f"集合 {self.collection_name} 已成功删除。")
                except MilvusException as e: self.logger.error(f"删除集合时出错: {e}", exc_info=True); raise
                except Exception as e: self.logger.error(f"删除集合时发生意外错误: {e}", exc_info=True); raise
            else: self.logger.info("删除操作已取消。")
        else: self.logger.info(f"集合 {self.collection_name} 不存在，无需删除。")

    def get_collection_stats(self):
        """获取集合的统计信息。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.warning(f"集合 {self.collection_name} 不存在，无法获取统计信息。"); return None
        self.logger.info(f"正在获取集合 '{self.collection_name}' 的统计信息...")
        try:
             stats_desc = self.collection.describe(); num_entities = self.collection.num_entities; is_loaded_flag = self.is_loaded(); indexes = self.collection.indexes
             self.logger.info(f"--- 集合 '{self.collection_name}' 统计信息 ---"); self.logger.info(f"  近似实体数量: {num_entities}"); self.logger.info(f"  是否已加载到内存: {'是' if is_loaded_flag else '否'}"); self.logger.debug(f"  Schema 详情: {stats_desc.get('fields')}"); self.logger.debug(f"  索引详情: {indexes}")
             return {"name": self.collection_name, "description": stats_desc.get('description'), "num_entities": num_entities, "is_loaded": is_loaded_flag, "schema": stats_desc.get('fields'), "indexes": [{"field": idx.field_name, "name": idx.index_name, "params": idx.params} for idx in indexes]}
        except MilvusException as e: self.logger.error(f"获取集合统计信息时出错: {e}", exc_info=True); return None
        except Exception as e: self.logger.error(f"获取集合统计信息时发生意外错误: {e}", exc_info=True); return None

    def delete_entities(self, expr, confirm_prompt=True):
        """根据 Milvus 表达式删除实体。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error("集合未初始化或加载失败，无法删除实体。"); raise ValueError("集合不可用。")
        if not expr or not isinstance(expr, str): self.logger.error("删除操作需要一个有效的 Milvus 表达式字符串。"); raise ValueError("无效的删除表达式。")
        self.logger.info(f"准备根据表达式 '{expr}' 删除实体..."); proceed = False
        try:
            self.logger.debug("正在查询匹配表达式的实体数量..."); pk_name = self.collection.schema.primary_field.name; count_result = self.collection.query(expr, output_fields=[f"count({pk_name})"]) # 使用 count(pk)
            match_count = 0
            if count_result and isinstance(count_result, list) and len(count_result) > 0: count_key = next(iter(count_result[0])); match_count = count_result[0].get(count_key, 0)
            self.logger.info(f"查询到 {match_count} 个实体匹配该表达式。")
            if match_count == 0: self.logger.info("没有匹配的实体，无需删除。"); return None
            if confirm_prompt: confirm = input(f"!!! 确认要删除这 {match_count} 个实体吗? (yes/no): "); proceed = (confirm.lower() == 'yes')
            else: proceed = True
            if proceed:
                self.logger.warning(f"正在执行删除操作，表达式: '{expr}'"); delete_result = self.collection.delete(expr); self.logger.info("删除请求已发送。正在执行 flush 操作..."); start_time = time.time(); self.collection.flush(); flush_time = time.time() - start_time # 删除后务必 flush
                self.logger.info(f"Flush 操作完成 (耗时 {flush_time:.2f} 秒)。已删除 {delete_result.delete_count} 条数据。")
                return delete_result
            else: self.logger.info("删除操作已取消。"); return None
        except MilvusException as e: self.logger.error(f"删除实体时出错 (表达式: '{expr}'): {e}", exc_info=True); raise
        except Exception as e: self.logger.error(f"删除实体时发生意外错误: {e}", exc_info=True); raise
    
    def delete_entities_by_field(self, field_name: str, field_value, confirm_prompt=True, auto_compact_threshold_mb=None):

        if self.collection is None:
            self._check_and_load_collection()
            if self.collection is None:
                self.logger.error(f"集合 {self.collection_name} 未创建或加载失败。无法删除实体。")
                raise ValueError(f"集合 {self.collection_name} 不可用。")

        # 检查字段是否存在于 Schema 中
        field_exists_in_schema = any(f.name == field_name for f in self.collection.schema.fields)
        if not field_exists_in_schema:
            self.logger.error(f"字段 '{field_name}' 不存在于集合 '{self.collection_name}' 的 Schema 中。无法执行删除。")
            raise ValueError(f"字段 '{field_name}' 无效。")

        # 构建删除表达式
        # 对于字符串类型的值，Milvus 表达式需要用引号包围
        if isinstance(field_value, str):
            if '"' in field_value and "'" in field_value:
                # 如果值同时包含单双引号，需要更复杂的转义，或 Milvus 可能不支持
                self.logger.error(f"字段值 '{field_value}' 同时包含单引号和双引号，可能导致表达式错误。")
                # 或者尝试替换一种引号，例如将双引号替换为 \" (如果Milvus支持)
                # sanitized_value = field_value.replace('"', '\\"') # 示例，可能不完美
                # expr = f'{field_name} == "{sanitized_value}"'
                raise ValueError(f"字段值 '{field_value}' 包含复杂引号，难以直接用于表达式。请手动构造表达式。")
            elif '"' in field_value:
                expr = f"{field_name} == '{field_value}'" # 使用单引号包围
            else:
                expr = f'{field_name} == "{field_value}"' # 默认使用双引号包围
        elif isinstance(field_value, (int, float, bool)):
            expr = f"{field_name} == {str(field_value).lower() if isinstance(field_value, bool) else field_value}"
        else:
            self.logger.error(f"字段值类型 {type(field_value)} 不支持直接用于表达式构建。请手动构造表达式或扩展此方法。")
            raise ValueError("不支持的字段值类型。")

        self.logger.info(f"准备根据表达式 '{expr}' 从集合 '{self.collection_name}' 删除实体...")
        
        proceed = False
        deleted_count_estimation = 0 # 用于判断是否需要 compact

        try:
            # 1. (可选) 查询匹配的实体数量以供确认
            if confirm_prompt or auto_compact_threshold_mb is not None:
                self.logger.debug(f"正在查询匹配表达式 '{expr}' 的实体数量...")
                # 为了获取数量，我们可以查询主键字段
                pk_field_name = self.collection.schema.primary_field.name
                # Milvus 2.2.9+ 支持 count(*)
                # count_result = self.collection.query(expr, output_fields=[f"count(*)"])
                # 对于旧版本，可以获取ID然后计数，但可能慢
                # 临时获取少量ID来估计，或直接获取所有ID（如果预期数量不多）
                # 为了简单，我们先假设可以直接获取数量，或提示用户
                # 注意: 直接查询大量数据可能很慢，count(*) 是 Milvus 2.2.9+ 的优化
                # 如果版本较低，获取数量可能需要变通，或跳过此精确计数步骤
                # 暂时我们假设可以直接获取
                try:
                    # 尝试使用 count(*), 如果失败则回退或只提示不显示数量
                    query_count_result = self.collection.query(expr=expr, output_fields=["count(*)"], timeout=10) # Milvus 2.2.9+
                    if query_count_result and isinstance(query_count_result, list) and len(query_count_result) > 0:
                        match_count = query_count_result[0].get('count(*)', 0)
                        deleted_count_estimation = match_count
                    else: # 回退或无法获取精确数量
                        self.logger.warning("无法通过 count(*) 获取精确匹配数量，将不显示计数或使用近似值。")
                        match_count = "未知数量的" # 替代方案
                except Exception as count_e:
                    self.logger.warning(f"查询匹配数量时出错 (可能不支持 count(*)): {count_e}。将不显示计数。")
                    match_count = "未知数量的"


                self.logger.info(f"查询到 {match_count} 个实体匹配该表达式。")
                if isinstance(match_count, int) and match_count == 0:
                    self.logger.info("没有匹配的实体，无需删除。")
                    return None

            if confirm_prompt:
                confirmation = input(f"!!! 警告: 您即将根据表达式 '{expr}' 从集合 '{self.collection_name}' 中删除 {match_count} 个实体。此操作可能无法轻易回滚。请输入 'yes' 确认删除: ")
                proceed = (confirmation.lower() == 'yes')
            else:
                proceed = True # 无需用户确认

            if proceed:
                self.logger.warning(f"正在执行删除操作，表达式: '{expr}'")
                delete_result = self.collection.delete(expr)
                self.logger.info(f"删除请求已发送。实际删除数量: {delete_result.delete_count}。")
                
                if delete_result.delete_count > 0:
                    self.logger.info("正在执行 flush 操作以持久化删除...")
                    start_time = time.time()
                    self.collection.flush()
                    flush_time = time.time() - start_time
                    self.logger.info(f"Flush 操作完成 (耗时 {flush_time:.2f} 秒)。")

                    # 检查是否需要自动 compact
                    if auto_compact_threshold_mb is not None and deleted_count_estimation > 0: # 确保有估计的删除数量
                        avg_vector_size_bytes = self.dimension * 4 # 4 bytes per float32
                        total_deleted_size_bytes = deleted_count_estimation * avg_vector_size_bytes
                        total_deleted_size_mb = total_deleted_size_bytes / (1024 * 1024)
                        self.logger.info(f"粗略估计已删除数据大小: {total_deleted_size_mb:.2f} MB (基于 {deleted_count_estimation} 个向量)。")

                        if total_deleted_size_mb >= auto_compact_threshold_mb:
                            self.logger.warning(f"已删除数据大小 ({total_deleted_size_mb:.2f} MB) 达到或超过阈值 ({auto_compact_threshold_mb} MB)。将尝试执行 compact 操作。")
                            self.compact_collection() # 调用 compact 方法
                        else:
                            self.logger.info("已删除数据大小未达到自动 compact 阈值。")
                else:
                    self.logger.info("没有实体被实际删除 (delete_result.delete_count 为 0)。")

                return delete_result
            else:
                self.logger.info("删除操作已取消。")
                return None
        except MilvusException as e:
            self.logger.error(f"删除实体时出错 (表达式: '{expr}'): {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"删除实体时发生意外错误: {e}", exc_info=True)
            raise

    def compact_collection(self):
        """
        对集合执行 compact 操作以回收已删除实体的物理空间。
        这是一个耗时的操作，并且是异步的。
        """
        if self.collection is None:
            self.logger.error(f"集合 {self.collection_name} 未初始化，无法执行 compact。")
            return False
        
        self.logger.warning(f"开始对集合 '{self.collection_name}' 执行 compact 操作。这可能需要较长时间，并且是异步的。")
        try:
            compaction_id = self.collection.compact()
            self.logger.info(f"Compact 操作已成功提交。Compaction ID: {compaction_id.compaction_id if hasattr(compaction_id, 'compaction_id') else compaction_id}")
            
            # 可以选择等待 compact 完成，但这可能会阻塞很长时间
            # self.wait_for_compaction_complete(compaction_id.compaction_id)
            return True
        except MilvusException as e:
            self.logger.error(f"提交 compact 操作时出错: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"提交 compact 操作时发生未知错误: {e}", exc_info=True)
            return False

    def get_compaction_state(self, compaction_id):
        """获取指定 compaction_id 的状态。"""
        if self.collection is None:
            self.logger.error("集合未初始化。"); return None
        try:
            state = self.collection.get_compaction_state(compaction_id=compaction_id)
            self.logger.info(f"Compaction ID {compaction_id} 的状态: {state.state if hasattr(state, 'state') else state}")
            return state
        except Exception as e:
            self.logger.error(f"获取 Compaction ID {compaction_id} 状态时出错: {e}"); return None

    def wait_for_compaction_complete(self, compaction_id, timeout_seconds=3600, poll_interval_seconds=30):
        """等待 compaction 操作完成。"""
        if self.collection is None:
            self.logger.error("集合未初始化。"); return False
        self.logger.info(f"正在等待 Compaction ID {compaction_id} 完成 (超时: {timeout_seconds}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                state = self.collection.get_compaction_state(compaction_id=compaction_id)
                # Milvus CompactionState: Unknown=0, Undefined=1, Executing=2, Completed=3
                if hasattr(state, 'state') and state.state == 3: # Completed
                    self.logger.info(f"Compaction ID {compaction_id} 已完成。")
                    return True
                elif hasattr(state, 'state') and state.state not in [0,1,2]: # 出现其他非执行中状态
                    self.logger.warning(f"Compaction ID {compaction_id} 进入非预期状态: {state.state}。停止等待。")
                    return False
                self.logger.debug(f"Compaction ID {compaction_id} 状态: {state.state if hasattr(state, 'state') else state} (仍在进行中...)")
            except Exception as e:
                self.logger.error(f"等待 compaction 完成期间查询状态出错: {e}"); return False # 查询状态失败，停止等待
            time.sleep(poll_interval_seconds)
        self.logger.warning(f"等待 Compaction ID {compaction_id} 超时 ({timeout_seconds}s)。")
        return False


    def query(self, expr, output_fields=None, limit=100):
        """执行非向量查询。"""
        if self.collection is None: self._check_and_load_collection();
        if self.collection is None: self.logger.error("集合未初始化或加载失败，无法执行查询。"); raise ValueError("集合不可用。")
        if not expr or not isinstance(expr, str): self.logger.error("查询操作需要一个有效的 Milvus 表达式字符串。"); raise ValueError("无效的查询表达式。")
        if output_fields is None: output_fields = [f.name for f in self.schema_fields if f.name != self.embedding_field_name]
        self.logger.info(f"执行查询: expr='{expr}', limit={limit}"); self.logger.debug(f"查询输出字段: {output_fields}")
        try:
             start_time = time.time(); results = self.collection.query(expr=expr, output_fields=output_fields, limit=limit, consistency_level="Strong", timeout=30); query_time = time.time() - start_time
             self.logger.info(f"查询完成 (耗时 {query_time:.3f} 秒)，返回 {len(results)} 条结果。")
             return results
        except MilvusException as e: self.logger.error(f"执行查询时出错 (表达式: '{expr}'): {e}", exc_info=True); return []
        except Exception as e: self.logger.error(f"执行查询时发生意外错误: {e}", exc_info=True); return []

    def clear_memory_resources(self):
        """尝试释放内存资源。"""
        self.logger.info("尝试清理内存资源..."); self.release_collection(); self.logger.info("正在执行 Python 垃圾回收..."); n = gc.collect(); self.logger.info(f"垃圾回收完成，清除了 {n} 个对象。")

    def disconnect(self):
        """断开与 Milvus 的连接。"""
        if self.connection_alias in connections.list_connections():
            self.logger.info(f"正在断开 Milvus 连接 (别名: {self.connection_alias})...")
            try: connections.disconnect(self.connection_alias); self.logger.info("连接已断开。")
            except Exception as e: self.logger.error(f"断开连接时出错: {e}", exc_info=True)
        else: self.logger.debug("连接别名不存在，无需断开。")
