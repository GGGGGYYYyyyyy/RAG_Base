from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_dir, device, use_fp16, query_instruction=None):
        self.device = device
        self.use_fp16 = use_fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        self.query_instruction = query_instruction or "为这个句子生成表示以用于检索相关文章："

    #bge的Embedding模型
    def generate_embeddings_bge(self, texts, max_length=512, batch_size=32, is_query=False):
        self.model.eval()
        max_length = min(max_length, self.tokenizer.model_max_length)
        embeddings = []

        if is_query:  # 添加指令仅用于查询
            texts = [f"{self.query_instruction}{t}" for t in texts]

        def batch_iter(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]

        for batch in batch_iter(texts, batch_size):
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                # 使用 CLS Token 向量并归一化
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)
    
    #对query添加指令版本的
    def encode_queries(self, queries, batch_size=32):
        """编码查询（自动添加指令）"""
        instructed_queries = [f"{self.query_instruction}{q}" for q in queries]
        return self.generate_embeddings_bge(instructed_queries, batch_size)

    def encode_documents(self, documents, batch_size=32):
        """编码文档（原始文本）"""
        return self.generate_embeddings_bge(documents, batch_size)
    
    #有道bce的Embedding模型
    def generate_embeddings_bce(self, texts, max_length=512):
        self.model.eval()
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()
    

    
# if __name__ == "__main__":
#     model_dir = "/home/yigao/RAG_cdipd_v2/BAAI-bge-large-zh-v1.5"  # 替换为模型的路径
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     use_fp16 = True  # 使用 FP16 可以加速计算，需 GPU 支持

#     # 初始化嵌入生成器
#     embedder = EmbeddingGenerator(model_dir, device, use_fp16)

#     # 示例查询和文档
#     queries = [
#         "如何优化检索系统的性能？",
#         "深度学习在自然语言处理中的应用有哪些？",
#     ]
#     documents = [
#         "检索系统的优化通常需要结合索引结构、排序算法以及硬件资源。",
#         "自然语言处理的主要任务包括机器翻译、文本摘要、情感分析等。",
#     ]

#     # 生成查询和文档的嵌入
#     print("正在生成查询嵌入...")
#     # query_embeddings = embedder.encode_queries(queries)
#     # print(f"查询嵌入完成，形状为：{query_embeddings.shape}")
#     query_embeddings  = embedder.encode_documents(queries)
#     print("正在生成文档嵌入...")
#     document_embeddings = embedder.encode_documents(documents)
#     print(f"文档嵌入完成，形状为：{document_embeddings.shape}")

#     # 示例：计算余弦相似度
#     def cosine_similarity(emb1, emb2):
#         emb1 = torch.tensor(emb1)
#         emb2 = torch.tensor(emb2)
#         similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
#         return similarity.numpy()

#     print("计算查询与文档的相似性：")
#     for i, query_embedding in enumerate(query_embeddings):
#         print(f"查询 {i + 1}：{queries[i]}")
#         for j, document_embedding in enumerate(document_embeddings):
#             similarity = cosine_similarity(query_embedding, document_embedding)
#             print(f"  文档 {j + 1} 相似度：{similarity:.4f}")