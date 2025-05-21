# RAG 智能问答系统 (RAG_Standard_V2)

本项目是一个基于检索增强生成 (Retrieval Augmented Generation, RAG) 范式的智能问答系统。它旨在通过结合先进的文档解析、高效的向量检索技术以及强大的大型语言模型 (LLM) 来提供精准、相关的答案。系统的核心创新在于其精细化的数据预处理流程和复杂的多阶段混合检索策略。

## ✨ 核心功能与创新点

### 1. 智能数据预处理与解析 (<mcfolder name="Parser"></mcfolder> 模块)

高质量的知识源是RAG系统性能的基石。本项目在数据预处理阶段采用了以下创新方法：

*   **增强型Markdown解析器功能** (核心实现于 `<mcsymbol name="EnhancedMarkdownParser" type="class"></mcsymbol>` 类):
    *   **智能分块 (Intelligent Chunking)**: 采用基于 Markdown 结构（如标题、段落、列表）和自定义分隔符的动态分块策略，确保知识片段的语义完整性，而非简单的固定大小切分。
    *   **标题继承与上下文关联**: 分块时能够自动关联最近的多级标题，为每个知识块补充更丰富的上下文信息，增强其独立可理解性。
    *   **表格内容转换**: 支持将 Markdown 或 HTML 表格转换为更易于 LLM 理解和检索的文本描述格式。
    *   **元数据集成**: 可选地从外部数据库 (如 MySQL) 拉取并关联元数据到文档块，丰富知识的结构化信息。

*   **文本预处理功能** (集成于 `<mcsymbol name="MilvusManager" type="class"></mcsymbol>` 类中的 `<mcsymbol name="TextPreprocessor" type="class"></mcsymbol>` 类):
    *   在文本送入 BM25 索引或进行某些分析前，进行标准化处理，如转小写、去除多余空格。
    *   支持停用词移除，以优化关键词检索的准确性。

### 2. 先进的多阶段混合检索策略 (<mcfolder name="Query"></mcfolder> 与 <mcfolder name="Milvus_DB"></mcfolder> 模块)

为了从海量知识中高效准确地召回最相关的上下文信息，系统采用了复杂且强大的混合检索机制。

#### 2.1 双知识库设计

系统同时维护两个主要的知识集合，均存储于 Milvus 向量数据库中：
*   **内容集合 (Content Collection)**: 存储从原始文档（如Markdown文件）解析出的、经过分块处理的知识片段。这是系统主要的知识来源。
*   **问题集合 (Questions Collection)**: 存储高质量的问答对。这些问答对主要通过 **<mcfolder name="Data_Enhance"></mcfolder> 模块**（例如，使用 `<mcsymbol name="generate_rag_questions" filename="add_question.py" type="function"></mcsymbol>` 函数并通过名为 `add_question` 的脚本调用）**利用大型语言模型 (LLM) 和精心设计的提示词 (Prompts) 针对内容集合中的每个知识片段自动生成**。这些提示词经过特别设计，旨在引导LLM从不同角度、针对不同信息点提出多样化且与原文紧密相关的问题。此外，问题集合也可以包含部分预先定义或从用户交互中积累的问答对。每个问答对通常包含一个或多个针对特定知识点的问题、以及指向内容集合中对应知识片段的引用 (如 `chunk_id`)。这种方式旨在为每个知识点生成多样化、高质量的潜在查询，从而提升检索的覆盖面和准确性。类似地，<mcfolder name="Data_Enhance"></mcfolder> 中的名为 `add_topic` 的脚本也使用LLM（通过 `<mcsymbol name="generate_rag_topic" filename="add_topic.py" type="function"></mcsymbol>` 函数）为知识片段生成核心主题句，进一步增强其可检索性。

#### 2.2 基础检索单元：混合检索

无论是对内容集合还是问题集合进行检索，系统都采用了一种混合检索方法，结合了语义向量检索和关键词检索的优势。此功能主要由 `<mcsymbol name="MilvusManager" type="class"></mcsymbol>` 类中的 `<mcsymbol name="hybrid_search" type="function"></mcsymbol>` 方法提供。

*   **向量检索 (Vector Search)**:
    *   **原理**: 利用文本嵌入模型 (如 `<mcsymbol name="EmbeddingGenerator" type="class"></mcsymbol>` 类) 将文本（查询文本和知识库中的文本块）转换为高维向量。这些向量能够捕捉文本的语义信息。
    *   **作用**: 在 Milvus 数据库中进行高效的语义相似度搜索，找出与用户查询在语义上最接近的知识片段或问题。即使措辞不同但含义相近，也能被召回。

*   **关键词检索 (BM25)**:
    *   **原理**: 一种经典的基于词频和逆文档频率的统计算法，用于评估文本片段与查询之间的相关性。它关注关键词的匹配度。
    *   **作用**: 弥补纯向量检索可能忽略关键词精确匹配的不足，确保包含重要关键词的文本片段能够被召回。BM25 排序在 `<mcsymbol name="MilvusManager" type="class"></mcsymbol>` 内部通过 `<mcsymbol name="_bm25_rank_docs" type="function"></mcsymbol>` 方法实现。

*   **结果融合 (Result Fusion)**:
    *   **目的**: 将向量检索和 BM25 检索各自召回的结果列表进行合并和重排序，以期获得比单一检索方法更优的综合结果。
    *   **策略**: 通常采用**倒数排序融合 (Reciprocal Rank Fusion, RRF)** 或加权融合。RRF 是一种无需调参的有效融合方法，它根据每个文档在不同检索结果列表中的排名来计算其最终得分。可调整 `rrf_k` (RRF融合时考虑的每个列表的文档数) 和不同检索方式的权重 (如 `bm25_weight`, `vector_weight`) 来优化融合效果。

#### 2.3 查询处理流程详解 (`<mcsymbol name="AdvancedQAQueryProcessor" type="class"></mcsymbol>` 类)

`<mcsymbol name="AdvancedQAQueryProcessor" type="class"></mcsymbol>` 类是整个复杂查询流程的编排者，其核心方法是 `<mcsymbol name="multi_stage_hybrid_search" type="function"></mcsymbol>`。该流程旨在最大化信息召回的全面性和准确性。

1.  **输入**:
    *   `query_text`: 用户提出的自然语言问题。
    *   可选参数: `top_k` (最终返回结果数量), `filters_expr_content` (内容集合的元数据过滤条件), `filters_expr_question` (问题集合的元数据过滤条件), 以及控制各阶段检索和融合行为的权重参数 (如 `content_vector_w`, `question_bm25_w` 等)。

2.  **查询向量化**:
    *   使用 <mcfolder name="Embedding"></mcfolder> 模块将用户输入的 `query_text` 转换为查询嵌入向量，供后续的向量检索使用。

3.  **并行检索阶段**:
    *   **内容集合检索**:
        *   使用查询嵌入向量和原始查询文本，在**内容集合**上执行上述的**混合检索** (`<mcsymbol name="hybrid_search" type="function"></mcsymbol>`)。
        *   召回一批与用户查询语义相关或关键词匹配的内容片段。
        *   此阶段的召回数量 (内部 `top_k`) 通常设置得较大 (如 `content_search_top_k_internal`)，以保证召回的全面性。
    *   **问题集合检索**:
        *   同样使用查询嵌入向量和原始查询文本，在**问题集合**上执行**混合检索**。
        *   召回一批与用户查询相似的、已存在的标准问题。
        *   此阶段的召回数量 (内部 `top_k`) 也设置得较大 (如 `question_search_top_k_internal`)。

4.  **上下文增强与初步融合**:
    *   **基于匹配问题的上下文提取**: 如果问题集合的检索结果中包含与用户查询高度相似的问题，系统会利用这些匹配问题所关联的 `chunk_id` (即它们答案来源的知识片段ID)。
    *   然后，根据这些 `chunk_id`，从**内容集合**中精确地、批量地获取对应的原始知识片段。这些通过“相似问题”间接获取的上下文，往往是高质量且高度相关的。
    *   **初步整合**: 将直接从内容集合检索到的片段，与通过问题集合间接获取的内容片段进行初步的整合和去重，形成一个更丰富的候选上下文池。

5.  **跨集合结果的最终融合**:
    *   **目的**: 将来自两个主要来源（直接内容检索和基于相似问题的内容检索）的候选上下文进行最终的、统一的排序。
    *   **策略**: 再次使用 RRF 或其他加权融合策略。此时，融合的对象是内容集合的直接检索结果列表和问题集合（及其增强上下文）的检索结果列表。
    *   **优势**: 这种跨集合融合能够充分利用两种知识源的优势：内容集合提供了广泛的知识覆盖，而问题集合则贡献了经过验证的高质量问答经验。通过融合，系统能够平衡直接内容匹配和历史问题匹配的重要性。
    *   可调整 `final_fusion_rrf_k` 等参数控制最终融合的力度。

6.  **输出**:
    *   经过上述多阶段检索和融合后，生成一个最终的、排序后的候选上下文列表。这个列表中的上下文片段被认为是与用户查询最相关的，将用于后续的答案生成。

#### 2.4 可选的重排序 (<mcfolder name="ReRank"></mcfolder> 模块)

在获得初步的候选上下文列表后，系统还支持引入一个独立的**重排序模型** (`<mcsymbol name="Reranker" type="class"></mcsymbol>` 类)。
*   **作用**: 重排序模型通常是更复杂的深度学习模型，它接收查询和候选上下文对，并对上下文的相关性进行更精细的打分和排序。
*   **时机**: 在 `<mcsymbol name="AdvancedQAQueryProcessor" type="class"></mcsymbol>` 完成其多阶段混合搜索之后，可以将得到的上下文列表传递给 `<mcsymbol name="Reranker" type="class"></mcsymbol>`。
*   **优势**: 能够进一步提升最终送给 LLM 的上下文质量，过滤掉部分在初步检索中可能存在的噪声或不完全相关的片段。

### 3. 模块化与可配置性

*   **清晰的模块划分**: 项目结构清晰，各功能模块（如 <mcfolder name="Parser"></mcfolder>, <mcfolder name="Embedding"></mcfolder>, <mcfolder name="Milvus_DB"></mcfolder>, <mcfolder name="Query"></mcfolder>, <mcfolder name="ReRank"></mcfolder>, <mcfolder name="LLM_Response"></mcfolder>, <mcfolder name="Data_Enhance"></mcfolder>）职责分明，易于理解和维护。
*   **配置驱动**: 核心参数（如模型标识、数据库连接信息、检索调优参数等）均通过集中的配置文件进行管理，方便调整和部署。
*   **数据增强工具 (<mcfolder name="Data_Enhance"></mcfolder> 模块)**: 提供了如名为 `add_question` 和 `add_topic` 的脚本，利用LLM为知识片段自动生成高质量的问题和主题句，极大丰富了知识库的可检索性，是提升系统性能的关键环节。

## 🚀 系统架构概览 (功能流)

```
用户查询 --> API服务 (接收与分发)
             |
             v
查询处理器 (AdvancedQAQueryProcessor) --[生成查询嵌入]--> Embedding模块
             |
             +--[混合检索]--> 内容知识库 (通过 MilvusManager 访问)
             |                 (向量搜索 + BM25)
             |
             +--[混合检索]--> 问题知识库 (通过 MilvusManager 访问)
             |                 (向量搜索 + BM25, 问题由LLM生成)
             |
             v
结果融合 (RRF等策略，多阶段) --> [可选] 重排序模块 (Reranker)
             |
             v
构建最终上下文 --> LLM响应模块 (LLMHandler，与大语言模型交互)
             |
             v
           智能答案
```

文档预处理与数据增强流程 (功能流):
```
原始文档 --> Parser模块 (EnhancedMarkdownParser)
              | (解析、智能分块、元数据关联)
              v
           结构化知识片段 --> Data_Enhance模块 (问题与主题生成脚本)
              |                            | (LLM生成问题、主题)
              |                            v
              |                       增强后的知识片段
              |
              v
Embedding模块 --[生成文本嵌入]--> Milvus数据导入脚本
                                 |
                                 v
                         向量数据库 (Milvus)
                         (内容集合、问题集合)
```

## 🛠️ 主要模块功能简介

*   **<mcfolder name="Parser"></mcfolder> 模块**: 负责将原始文档（特别是Markdown）解析、清洗、并进行智能分块，提取元数据，为后续的嵌入和检索做准备。核心是 `<mcsymbol name="EnhancedMarkdownParser" type="class"></mcsymbol>` 类。
*   **<mcfolder name="Embedding"></mcfolder> 模块**: 负责将文本数据转换为向量嵌入。核心是 `<mcsymbol name="EmbeddingGenerator" type="class"></mcsymbol>` 类，支持加载和使用各种预训练的嵌入模型。
*   **<mcfolder name="Milvus_DB"></mcfolder> 模块**: 封装了与 Milvus 向量数据库的所有交互。核心是 `<mcsymbol name="MilvusManager" type="class"></mcsymbol>` 类，提供集合创建、数据增删改查、向量搜索、BM25搜索及混合搜索等功能。一个专门的脚本用于初始化数据库和批量导入数据。
*   **<mcfolder name="Query"></mcfolder> 模块**: 实现复杂查询处理逻辑的核心。`<mcsymbol name="AdvancedQAQueryProcessor" type="class"></mcsymbol>` 类编排了多阶段、跨集合的混合检索流程。
*   **<mcfolder name="ReRank"></mcfolder> 模块**: 提供重排序功能，使用专门的重排序模型对初步检索到的上下文进行二次精排。核心是 `<mcsymbol name="Reranker" type="class"></mcsymbol>` 类。
*   **<mcfolder name="LLM_Response"></mcfolder> 模块**: 负责与大语言模型 (LLM) 进行交互。`<mcsymbol name="LLMHandler" type="class"></mcsymbol>` 类构建发送给 LLM 的 Prompt (包含查询和检索到的上下文)，并处理 LLM 返回的响应，支持流式和非流式输出。
*   **<mcfolder name="Data_Enhance"></mcfolder> 模块**: 包含用于数据增强的辅助脚本。核心功能是利用LLM和精心设计的提示词，为每个知识片段自动生成多样化的问题（通过名为 `add_question` 的脚本中的 `<mcsymbol name="generate_rag_questions" type="function"></mcsymbol>` 函数）和核心主题句（通过名为 `add_topic` 的脚本中的 `<mcsymbol name="generate_rag_topic" type="function"></mcsymbol>` 函数），这些生成的内容会被添加到知识片段的元数据中，并用于构建问题知识库，从而显著提升检索效果。
*   **<mcfolder name="Config"></mcfolder> 模块**: 集中管理项目的所有配置信息，如模型标识、API密钥、数据库地址、检索参数等。
*   **API 服务**: 基于 FastAPI 构建的Web服务入口，对外提供问答API接口，接收用户请求，调用后端服务，并返回结果。

## ⚙️ 运行与使用简述

1.  **环境准备**: 安装 Python 及所有必要的依赖库 (通常通过项目依赖文件)。确保 Milvus 服务已启动并可访问。
2.  **配置系统**: 修改项目中的配置文件，指定正确的模型设置、Milvus 服务地址、LLM API 信息等。
3.  **数据处理与入库**:
    *   使用 <mcfolder name="Parser"></mcfolder> 模块处理您的原始文档，生成结构化的数据。
    *   （关键步骤）运行 <mcfolder name="Data_Enhance"></mcfolder> 模块中的脚本（如问题生成脚本、主题生成脚本），为解析后的数据生成问题和主题。
    *   运行数据导入脚本，将处理并增强后的数据及其向量嵌入导入到 Milvus 数据库中。
4.  **启动服务**: 运行 API 服务的主程序文件 (通常使用 Uvicorn)。
5.  **发起查询**: 通过 API 接口 (如 `/query`) 发送查询请求。

## 🤝 贡献

欢迎对本项目提出改进意见或贡献代码。

## 📄 许可证
若要使用请联系作者
