# 导入所需的库
import requests  # 用于发送 HTTP 请求
import json      # 用于处理 JSON 数据
import os        # 用于与操作系统交互，如文件路径操作、检查文件/目录存在性
import argparse  # 用于解析命令行参数
import time      # 用于在重试时暂停
import logging   # 使用日志记录替代部分 print

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VLLM_API_URL = "xxx"
MODEL_NAME = "QwQ32B"
END_THINK_TAG = "</think>" 

def generate_rag_topic(content: str, api_url: str = VLLM_API_URL, model_name: str = MODEL_NAME) -> str | None:
    """
    使用本地 LLM (通过 vLLM API) 为给定的文本内容生成一个或多个主题句。
    该函数会处理并移除 LLM 可能产生的 <think>...</think> 块。

    Args:
        content (str): 讲话内容的文本块 (Chunk)。
        api_url (str): vLLM 聊天完成端点的 URL。
        model_name (str): vLLM 服务器上部署的模型名称。

    Returns:
        str | None: 清理后的主题句列表字符串，如果发生错误则返回包含错误信息的占位符主题。
    """
    # 检查内容是否为空或仅包含空白字符
    if not content or not content.strip():
        logger.warning("内容为空，无法生成主题。")
        return "[无内容]" # 为空内容返回一个占位符主题

    # --- 为 LLM 构建提示 (Prompt) ---
    # (保持和你之前提供的最新 Prompt 一致)
    prompt = f"""
    # 角色设定
    你是一位专业的文本分析专家，擅长从政xxx的**文本片段 (Chunk)** 中快速提炼核心信息点。

    # 任务描述
    你的任务是仔细阅读并理解以下提供的xxx的**一个文本块 (Chunk，约 512 token)**。基于对**当前文本块内容**的分析，提取出能够代表**该块主要信息**的**核心主题句 (Theme Sentences)** 列表。这些主题句将用于后续的 RAG 检索，帮助定位到包含相关信息的具体文本块。

    # 核心要求
    1.  **聚焦当前 Chunk：** 生成的主题句必须**严格、完全**基于当前提供的这个文本块的内容。不要假设或引入任何来自该文本块之外的信息。
    2.  **服务于RAG检索：** 主题句应能概括**当前文本块**的关键信息点，帮助用户通过检索快速定位到这个具体的 Chunk。
    3.  **主题形式 (句子)：** 每个主题应为一个**简洁的陈述句**，清晰概括**当前文本块内**的一个核心议题、行动、成果或观点。
    4.  **精准反映片段：** 主题句必须准确反映**当前文本块**所阐述的具体内容。
    5.  **覆盖关键点 (Chunk内)：** 提取的主题句列表应尽可能覆盖**当前文本块中**的所有主要论点和重要信息点。
    6.  **避免过度泛化：** 即使是句子，也要避免对**当前文本块内容**进行过于笼统的概括。
    7.  **数量适中：** 根据**当前文本块**的内容复杂度，提取适量的核心主题句（通常 1-5 个即可，取决于块内信息密度）。

    # 输出格式
    请严格按照以下格式输出，**仅输出主题句列表**，每个主题句占一行，使用无序列表（例如，使用减号 `-` 或星号 `*`）。不要包含任何其他说明性文字、标题或解释。

    # 示例 (One-Shot Example - 基于片段提取句子主题)

    ---
    **示例输入文本 (假设这是一个 Chunk):**
    xxx
    ---
    **示例输出主题 (句子形式 - 基于该片段):**
    - xxx
    ---

    # 待分析的xxx片段 (Chunk)
    ---
    {content}
    ---

    # 开始执行
    请根据以上要求和**句子形式的示例**，对上面 **待分析的领导讲话稿片段 (Chunk)** 进行分析，并生成核心主题句列表。请**严格聚焦于当前提供的文本块内容**。
    """

    # --- 准备 vLLM API 的请求体 (Payload) ---
    payload = {
        "model": model_name, # 使用配置的模型名称
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000, # 允许模型生成较长的思考过程和最终结果
        "temperature": 0.3,
        "top_p": 0.9,
    }
    headers = {
        "Content-Type": "application/json"
    }
    max_retries = 2
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            logger.info(f"  [LLM 请求尝试 {attempt+1}/{max_retries}] 正在请求生成主题...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=90) # 超时时间90秒
            response.raise_for_status()
            result = response.json()

            # --- 提取并清理生成的文本 ---
            if "choices" in result and len(result["choices"]) > 0:
                # 1. 获取完整的原始响应内容
                full_response_content = result["choices"][0].get("message", {}).get("content", "").strip()
                logger.debug(f"  LLM 原始响应: {full_response_content[:200]}...") # 打印前200字符以供调试

                # 2. 处理 <think> 标签
                content_for_topic = "" # 初始化用于存储最终内容的变量
                tag_end_pos = full_response_content.find(END_THINK_TAG)

                if tag_end_pos != -1:
                    # 如果找到 </think> 标签，只取它之后的内容
                    content_for_topic = full_response_content[tag_end_pos + len(END_THINK_TAG):].strip()
                    logger.info(f"  检测到并移除了 <think> 块。使用后续内容。")
                else:
                    # 如果未找到标签，使用完整的响应内容
                    content_for_topic = full_response_content
                    logger.debug(f"  未在响应中检测到 <think> 块。")

                #3. （可选）移除其他常见前缀 (如果需要)
                prefixes_to_remove = ["生成的主题:", "主题:", "核心主题句:", "主题句列表:"]
                for prefix in prefixes_to_remove:
                    if content_for_topic.startswith(prefix):
                        content_for_topic = content_for_topic[len(prefix):].strip()
                        logger.debug(f"  移除了前缀 '{prefix}'")
                        break

                # 4. 简单验证清理后的结果
                if not content_for_topic or len(content_for_topic) < 5: # 稍微增加长度检查
                     logger.warning(f"  警告：清理后收到可能无效的主题: '{content_for_topic}'。使用占位符。")
                     return "[主题生成结果无效]"

                # 成功生成并清理，返回主题
                logger.info(f"  成功生成并清理主题:\n{content_for_topic}")
                return content_for_topic # 返回清理后的内容
            else:
                logger.error(f"  错误：在第 {attempt+1} 次尝试中，LLM API 返回了预期之外的响应格式。响应: {result}")

        except requests.exceptions.Timeout:
             logger.error(f"  错误：LLM API 请求在第 {attempt+1} 次尝试时超时。")
             if attempt == max_retries - 1:
                 logger.error("  已达到最大超时重试次数。")
                 return "[主题生成超时]"
             logger.info(f"  将在 {retry_delay} 秒后重试...")
             time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"  错误：在第 {attempt+1} 次尝试连接 LLM API ({api_url}) 时出错: {e}")
            if attempt == max_retries - 1:
                 logger.error("  已达到最大连接错误重试次数。")
                 return "[主题生成API错误]"
            logger.info(f"  将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        except json.JSONDecodeError as e:
             logger.error(f"  错误：解码来自 LLM API 的 JSON 响应时出错: {e}")
             logger.error(f"  响应文本: {response.text if 'response' in locals() else 'N/A'}")
             return "[主题生成响应格式错误]"
        except Exception as e:
            logger.exception(f"  在 LLM 调用期间发生意外错误: {e}") # 使用 exception 记录堆栈跟踪
            return "[主题生成未知错误]"

    # 如果所有重试都失败了
    logger.error("  错误：生成主题的所有尝试均失败。")
    return "[主题生成失败]"

def process_json_file(input_filepath: str, output_dir: str):
    """
    读取一个 JSON 文件，为其中的每个（或单个）讲话片段添加 'topic' 字段 (通过 LLM 生成，处理 <think> 标签)，
    并将修改后的数据保存到输出目录。

    Args:
        input_filepath (str): 输入的 JSON 文件的完整路径。
        output_dir (str): 保存处理后文件的目标目录路径。
    """
    logger.info(f"\n开始处理文件: {input_filepath}")
    filename = os.path.basename(input_filepath)
    output_filepath = os.path.join(output_dir, filename)

    # 2. 读取 JSON 数据
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

    # 3. 处理讲话片段
    processed_count = 0
    total_snippets = 0

    if isinstance(data, list):
        total_snippets = len(data)
        logger.info(f"  找到包含 {total_snippets} 个片段的列表。")
        for i, snippet in enumerate(data):
            if isinstance(snippet, dict) and "content" in snippet:
                logger.info(f"  正在处理片段 {i+1}/{total_snippets}...")
                content = snippet.get("content", "")
                topic = generate_rag_topic(content) # 调用包含<think>处理的函数
                snippet["topic"] = topic
                if topic and not topic.startswith("[") and not topic.endswith("]"):
                     processed_count += 1
            else:
                logger.warning(f"  警告：列表中的第 {i+1} 项不是包含 'content' 的有效片段字典。已跳过。")
    elif isinstance(data, dict) and "content" in data:
        total_snippets = 1
        logger.info("  找到单个片段对象。")
        content = data.get("content", "")
        topic = generate_rag_topic(content) # 调用包含<think>处理的函数
        data["topic"] = topic
        if topic and not topic.startswith("[") and not topic.endswith("]"):
              processed_count += 1
    else:
        logger.error(f"  错误：文件 '{filename}' 包含未知的 JSON 结构。期望是包含片段的列表或单个片段对象（需含 'content' 字段）。")
        return

    # 4. 将修改后的数据保存到输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"  成功处理 {processed_count}/{total_snippets} 个片段。")
        logger.info(f"  已将处理后的文件保存至: {output_filepath}")
    except IOError as e:
        logger.error(f"  错误：无法写入输出文件: {output_filepath}。错误: {e}")
    except Exception as e:
        logger.error(f"  保存文件 {output_filepath} 时出错: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 格式的讲话稿片段，使用 LLM 添加主题字段 (处理 <think> 标签)。")
    parser.add_argument("input_path", help="输入的 JSON 文件路径或包含 JSON 文件的目录路径。")
    parser.add_argument("output_dir", help="用于保存处理后文件的目录路径。")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志记录级别 (默认: INFO)")

    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir

    # 设置日志级别
    logger.setLevel(args.log_level.upper())
    # 可以在这里添加 FileHandler 来将日志写入文件

    # 路径有效性检查
    if not os.path.exists(input_path):
        logger.critical(f"错误：输入路径不存在: {input_path}")
        exit(1)

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.critical(f"错误：无法创建输出目录: {output_dir}。错误: {e}")
        exit(1)

    # 根据输入路径类型进行处理
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
