# 导入所需的库
import requests
import json
import os
import argparse
import time
import logging
from config.config import(VLLM_URL, MODEL_NAME)
# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
VLLM_API_URL = VLLM_URL
MODEL_NAME = MODEL_NAME
END_THINK_TAG = "</think>"

def generate_rag_questions(content: str, origin_file_name: str, section_title: str, api_url: str = VLLM_API_URL, model_name: str = MODEL_NAME) -> str | None:
    """
    使用本地 LLM (通过 vLLM API) 为给定的文本内容生成一组多样化的、
    直接针对内容的问题，旨在最大化检索命中率。
    该函数会处理并移除 LLM 可能产生的 <think>...</think> 块，并结合提供的文档完整标题 (origin_file_name) 和章节标题信息。
    包含系统提示词。输入内容中的花括号会被转义。
    """
    if not content or not content.strip():
        logger.warning("内容为空，无法生成问题。")
        return "[无内容]"

    # --- 为 LLM 构建提示 (Prompt) ---

    # 1. 定义系统提示词 (保持不变，因为它描述了 originFileName 的用途)
    system_prompt_content = """
    你是xxx解读专家，特别擅长从xxx的片段 (Chunk) 中，提炼出能够被用户通过各种自然语言查询方式找到该片段的核心问题。你的目标是生成一系列能够最大化检索命中率的问题。
    你必须严格遵循用户提供的后续指示，包括输出格式、问题数量和具体要求。
    你将结合用户提供的文档完整标题（originFileName，通常包含标准号和标准名称）和章节/条款标题（section_title）来增强问题的上下文相关性。
    你生成的每个问题都必须能从用户提供的当前文本块 (Chunk) 中找到明确、直接的答案。
    输出的问题列表应使用减号 `-` 或星号 `*` 作为无序列表项，每个问题占一行，不包含任何额外的解释或前缀。
    """

    # 2. 定义用户提示词模板 (修改占位符)
    user_prompt_template = """
    # 任务描述
    仔细阅读并理解以下提供的**文档标题、章节/条款标题**和**一个文本块 (Chunk)**。基于对**当前文本块核心内容**的分析，并结合提供的文档和章节/条款标题，生成**一组（3-7个，根据信息密度调整）高度相关且多样化的问题**。这些问题应该：
    1.  直接指向文本块中的具体信息点，如定义、要求、标准值、程序步骤、适用范围、目的等。
    2.  尽可能从不同角度提问，特别关注：
        *   **定义类**：XX是什么？XX如何定义？
        *   **规范/要求类**：XX应符合什么标准？XX有哪些要求？XX必须如何操作？
        *   **条件/范围类**：在什么情况下适用XX？XX的适用范围是什么？
        *   **目的/原则类**：制定XX的目的是什么？XX遵循什么原则？
        *   **具体数值/指标类**：XX的具体参数/限值是多少？
        *   **程序/方法类**：如何执行XX？XX的流程是怎样的？
    3.  覆盖文本块中的主要术语、规定、指标、流程和原则。
    4.  模拟用户在查阅标准、法规时可能提出的具体、专业的自然语言问句。

    # 核心要求
    1.  **利用提供的上下文信息：** 在生成问题时，**必须**优先使用并结合提供的**文档完整标题（originFileName，通常包含标准号和名称）**和**章节/条款标题（section_title）**，使问题更具指向性。例如，可以问“根据《[originFileName的值]》中关于[章节/条款标题]的规定，...？”
    2.  **聚焦当前 Chunk：** 生成的问题的所有答案都必须**严格、完全**基于当前提供的这个文本块的内容。
    3.  **最大化检索覆盖：**
        *   多样性：针对同一核心信息，尝试用不同的问法或关注点来提问。
        *   全面性：尽可能覆盖文本块中的所有关键规定和信息点。
        *   具体性：问题应尽可能具体，避免泛泛而谈。
    4.  **模拟专业用户查询：** 问题应使用该领域用户可能使用的、相对规范和专业的语言。
    5.  **数量灵活：** 根据当前文本块的信息密度，生成 3-7 个问题。

 
    ---
    **示例输入 (规划标准类)：**
    文档完整标题 (originFileName): 《xxx》 
    章节标题 (section_title): 3.3 防洪及内涝整治
    文本片段 (Chunk):
    ....
    ---
    **示例输出问题：**
    -填充
    ---
    # 待分析的文档信息和文本块
    ---
    文档完整标题 (originFileName): {origin_file_name} 
    章节标题 (section_title): {section_title}
    文本片段 (Chunk):
    {content}
    ---
    # 开始执行
    请根据以上要求和示例，对上面提供的文本块进行分析，并生成问题列表。
"""
    # 3. 转义输入内容中的花括号 (变量名与函数参数一致)
    safe_origin_file_name_param = str(origin_file_name if origin_file_name else "未知文档").replace("{", "{{").replace("}", "}}")
    safe_section_title_param = str(section_title if section_title else "未知章节").replace("{", "{{").replace("}", "}}")
    safe_content_param = str(content if content else "").replace("{", "{{").replace("}", "}}")

    # 4. 格式化用户提示词 (关键字参数与模板占位符一致)
    try:
        formatted_user_prompt = user_prompt_template.format(
            origin_file_name=safe_origin_file_name_param, # <--- 修改关键字参数
            section_title=safe_section_title_param,
            content=safe_content_param
        )

    except KeyError as e:
        logger.error(f"格式化用户提示词时发生 KeyError: {e}")
        logger.error(f"请仔细检查 user_prompt_template 中的占位符是否都已在 .format() 中正确提供，并且名称无误。")
        logger.error(f"User Prompt Template (raw for checking placeholders):\n{user_prompt_template}")
        return "[问题生成模板格式化错误]"
    except Exception as e:
        logger.error(f"格式化用户提示词时发生未知错误: {e}")
        return "[问题生成模板格式化未知错误]"

    # 5. 构建 payload (与之前相同)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": formatted_user_prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    headers = {
        "Content-Type": "application/json"
    }
    max_retries = 2
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            logger.info(f"  [LLM 请求尝试 {attempt+1}/{max_retries}] 正在为文档 '{origin_file_name}' (章节: '{section_title}') 生成问题...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=100)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                full_response_content = result["choices"][0].get("message", {}).get("content", "").strip()
                logger.debug(f"  LLM 原始响应: {full_response_content[:250]}...")

                content_for_questions = ""
                tag_end_pos = full_response_content.find(END_THINK_TAG)

                if tag_end_pos != -1:
                    content_for_questions = full_response_content[tag_end_pos + len(END_THINK_TAG):].strip()
                    logger.info(f"  检测到并移除了 <think> 块。使用后续内容。")
                else:
                    content_for_questions = full_response_content
                    logger.debug(f"  未在响应中检测到 <think> 块。")

                prefixes_to_remove = [
                    "生成的问题:", "问题列表:", "问题:", "核心问题:", "以下是生成的问题:",
                    "以下是结合上下文生成的问题:", "根据提供的文本和标题，生成的问题如下:",
                    "以下是根据提供的标题和文本片段生成的问题列表:",
                    "问题列表 (结合文档标题和章节标题):",
                    "以下是旨在最大化检索命中率的问题："
                ]
                for prefix in prefixes_to_remove:
                    if content_for_questions.startswith(prefix):
                        content_for_questions = content_for_questions[len(prefix):].strip()
                        logger.debug(f"  移除了前缀 '{prefix}'")
                        break
                
                if not content_for_questions or len(content_for_questions) < 10: # 增加长度判断
                     logger.warning(f"  警告：清理后收到可能无效的问题列表: '{content_for_questions}'。使用占位符。")
                     return "[问题生成结果无效]"

                logger.info(f"  成功生成并清理检索优化型问题:\n{content_for_questions}")
                return content_for_questions
            else:
                logger.error(f"  错误：在第 {attempt+1} 次尝试中，LLM API 返回了预期之外的响应格式。响应: {result}")

        except requests.exceptions.Timeout:
             logger.error(f"  错误：LLM API 请求在第 {attempt+1} 次尝试时超时。")
             if attempt == max_retries - 1:
                 logger.error("  已达到最大超时重试次数。")
                 return "[问题生成超时]"
             logger.info(f"  将在 {retry_delay} 秒后重试...")
             time.sleep(retry_delay)
        except requests.exceptions.HTTPError as e:
            logger.error(f"  错误：LLM API 请求在第 {attempt+1} 次尝试时返回HTTP错误: {e}. 响应: {e.response.text if e.response else 'No response body'}")
            if attempt == max_retries - 1:
                 logger.error("  已达到最大HTTP错误重试次数。")
                 return "[问题生成API HTTP错误]"
            logger.info(f"  将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"  错误：在第 {attempt+1} 次尝试连接 LLM API ({api_url}) 时出错: {e}")
            if attempt == max_retries - 1:
                 logger.error("  已达到最大连接错误重试次数。")
                 return "[问题生成API连接错误]"
            logger.info(f"  将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        except json.JSONDecodeError as e:
             logger.error(f"  错误：解码来自 LLM API 的 JSON 响应时出错: {e}")
             logger.error(f"  响应文本: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
             return "[问题生成响应格式错误]"
        except Exception as e:
            logger.error(f"  在 LLM 调用或准备阶段发生意外错误: {e}")
            logger.exception("  详细错误信息:")
            return "[问题生成未知错误]"

    logger.error("  错误：生成问题的所有尝试均失败。")
    return "[问题生成失败]"

def process_json_file(input_filepath: str, output_dir: str):
    """
    读取一个 JSON 文件，为其中的每个片段添加 'questions' 字段 (通过 LLM 生成，使用 metadata.originFileName 作为文档标题)。
    """
    logger.info(f"\n开始处理文件: {input_filepath}")
    filename = os.path.basename(input_filepath)
    output_filepath = os.path.join(output_dir, filename)

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"  错误：输入文件未找到: {input_filepath}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"  错误：解析文件中的 JSON 失败: {input_filepath}。错误: {e}")
        return
    except Exception as e:
        logger.error(f"  读取文件 {input_filepath} 时出错: {e}")
        return

    processed_count = 0
    total_snippets = 0

    # 确定数据是列表还是单个字典
    snippets_to_process = []
    if isinstance(data, list):
        snippets_to_process = data
        total_snippets = len(data)
        logger.info(f"  找到包含 {total_snippets} 个片段的列表。")
    elif isinstance(data, dict):
        snippets_to_process = [data] # 将单个字典包装在列表中以便统一处理
        total_snippets = 1
        logger.info("  找到单个片段对象。")
    else:
        logger.error(f"  错误：文件 '{filename}' 包含未知的 JSON 结构。期望是包含片段的列表或单个片段对象。")
        return

    for i, snippet in enumerate(snippets_to_process):
        if isinstance(snippet, dict) and "content" in snippet and "metadata" in snippet:
            logger.info(f"  正在处理片段 {i+1}/{total_snippets}...")
            content = snippet.get("content", "")
            metadata = snippet.get("metadata", {})
            
            # 使用 originFileName 作为文档标题
            origin_file_name = metadata.get("originFileName", metadata.get("doc_title", "未知文档")) # 如果没有originFileName，回退到doc_title
            if not origin_file_name: # 再次检查，确保不为空
                origin_file_name = "未知文档"
                logger.warning(f"    片段 {i+1}/{total_snippets} 的 metadata 中缺少 'originFileName' 和 'doc_title'，将使用 '未知文档' 作为标题。")

            section_title = metadata.get("section_title", "未知章节")
            if not section_title:
                section_title = "未知章节"
                logger.warning(f"    片段 {i+1}/{total_snippets} 的 metadata 中缺少 'section_title'，将使用 '未知章节'。")


            questions = generate_rag_questions(content, origin_file_name, section_title)
            snippet["questions"] = questions
            if questions and not questions.startswith("[") and not questions.endswith("]"): # 检查是否是错误占位符
                 processed_count += 1
            else:
                 logger.warning(f"  片段 {i+1}/{total_snippets} (文档: '{origin_file_name}') 未能成功生成有效问题，结果为: {questions}")
        else:
            logger.warning(f"  警告：列表中的第 {i+1} 项不是包含 'content' 和 'metadata' 的有效片段字典。已跳过。")

    # 保存修改后的数据 (如果数据是单个字典，则保存该字典；如果是列表，则保存列表)
    data_to_save = data # data 变量仍然指向原始加载的列表或字典结构
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logger.info(f"  成功为 {processed_count}/{total_snippets} 个片段生成了问题（可能包含占位符）。")
        logger.info(f"  已将处理后的文件保存至: {output_filepath}")
    except IOError as e:
        logger.error(f"  错误：无法写入输出文件: {output_filepath}。错误: {e}")
    except Exception as e:
        logger.error(f"  保存文件 {output_filepath} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 格式的文档片段，使用 LLM 添加 'questions' 字段 (使用 metadata.originFileName 作为文档标题)。")
    parser.add_argument("input_path", help="输入的 JSON 文件路径或包含 JSON 文件的目录路径。")
    parser.add_argument("output_dir", help="用于保存处理后文件的目录路径。")
    parser.add_argument("--api-url", default=VLLM_API_URL, help=f"vLLM API 端点 (默认: {VLLM_API_URL})")
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"vLLM 模型名称 (默认: {MODEL_NAME})")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志记录级别 (默认: INFO)")

    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    
    VLLM_API_URL = args.api_url # 更新全局变量
    MODEL_NAME = args.model_name # 更新全局变量

    logger.setLevel(args.log_level.upper())

    if not os.path.exists(input_path):
        logger.critical(f"错误：输入路径不存在: {input_path}")
        exit(1)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.critical(f"错误：无法创建输出目录: {output_dir}。错误: {e}")
        exit(1)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".json"):
            process_json_file(input_path, output_dir)
        else:
            logger.error(f"错误：输入文件 '{input_path}' 不是一个 .json 文件。")
            exit(1)
    elif os.path.isdir(input_path):
        logger.info(f"开始处理目录 '{input_path}' 中的所有 .json 文件...")
        found_json = False
        json_files = [f for f in os.listdir(input_path) if f.lower().endswith(".json")]
        if json_files:
            found_json = True
            logger.info(f"找到 {len(json_files)} 个 JSON 文件。")
            for filename in json_files:
                filepath = os.path.join(input_path, filename)
                process_json_file(filepath, output_dir)
        if not found_json:
             logger.warning(f"警告：在目录 '{input_path}' 中未找到任何 .json 文件。")
    else:
        logger.critical(f"错误：输入路径既不是有效的文件也不是有效的目录: {input_path}")
        exit(1)

    logger.info("\n处理完成。")

#python add_question.py json文件地址 输出的json文件地址