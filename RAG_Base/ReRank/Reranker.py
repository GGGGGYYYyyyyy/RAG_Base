import torch
class Reranker:
    def __init__(self, reranker_model, tokenizer, device="cuda"):
        """
        初始化 Reranker 类。
        
        :param reranker_model: 用于重新排序的模型。
        :param tokenizer: 与 reranker_model 配套的分词器。
        :param device: 指定运行模型的设备（默认 "cuda"）。
        """
        self.reranker_model = reranker_model
        self.tokenizer = tokenizer
        self.device = device
        self.reranker_model.to(self.device)
        self.reranker_model.eval()  # 设置模型为评估模式

    # 当model为BAAI/bge-reranker-v2-m3时重排序的格式
    def rerank_bge(self, results, query):
        """
        使用BGE重排序模型对结果进行重排序
        
        :param results: QueryProcessor返回的结果列表
        :param query: 查询文本
        :return: 重排序后的结果列表
        """
        # 构建模型输入的 pairs
        pairs = [
            (query, res["Content"]) for res in results
        ]
        
        # 准备输入数据
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理获取分数
        with torch.no_grad():
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1).float()
        
        # 更新结果列表并根据分数排序
        for i, res in enumerate(results):
            res["Rerank Score"] = scores[i].item()
        return sorted(results, key=lambda x: x["Rerank Score"], reverse=True)
    
    # 当model为有道bce时候的重排序格式
    def rerank_bce(self, results, query):
        """
        使用BCE重排序模型对结果进行重排序
        
        :param results: QueryProcessor返回的结果列表
        :param query: 查询文本
        :return: 重排序后的结果列表
        """
        # 准备输入数据
        inputs = self.tokenizer(
            [(query, res["Content"]) for res in results],
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理获取分数
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze(-1)
        
        # 更新结果列表并根据分数排序
        for i, res in enumerate(results):
            res["Rerank Score"] = scores[i].item()
        return sorted(results, key=lambda x: x["Rerank Score"], reverse=True)
    
    def rerank(self, results, query, model_type="bge",top_k=None):
        """
        根据模型类型选择合适的重排序方法
        
        :param results: QueryProcessor返回的结果列表
        :param query: 查询文本
        :param model_type: 模型类型，"bge"或"bce"
        :return: 重排序后的结果列表
        """
        if model_type.lower() == "bge":
            reranked_results = self.rerank_bge(results, query)
            # return self.rerank_bge(results, query)
        elif model_type.lower() == "bce":
            reranked_results = self.rerank_bce(results, query)
            # return self.rerank_bce(results, query)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        if top_k is not None and top_k > 0 and top_k < len(reranked_results):
            return reranked_results[:top_k]
        
        return reranked_results
    
    def batch_rerank(self, batch_results, queries, model_type="bge",top_k=None):
        """
        批量重排序多个查询的结果
        
        :param batch_results: 多个查询的结果列表的列表
        :param queries: 对应的查询文本列表
        :param model_type: 模型类型，"bge"或"bce"
        :return: 重排序后的多个查询结果列表的列表
        """
        if len(batch_results) != len(queries):
            raise ValueError("批量结果数量与查询数量不匹配")
            
        reranked_results = []
        for i, (results, query) in enumerate(zip(batch_results, queries)):
            reranked_results.append(self.rerank(results, query, model_type,top_k))
            
        return reranked_results