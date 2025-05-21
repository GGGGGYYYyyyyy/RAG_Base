import os
import json
import re
import logging
from typing import Any, List, Set, Union

# --- 配置 ---
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- 核心清理函数 (与移除所有换行符版本相同) ---

def clean_json_recursive(data: Any, target_keys: Set[str]) -> Any:
    """
    递归地遍历 Python 对象（来自加载的 JSON），并从与目标键关联的字符串值中
    *完全移除*所有换行符 (\n)。
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            if isinstance(value, str) and key in target_keys:
                cleaned_value = value.replace('\n', '').strip()
                cleaned_dict[key] = cleaned_value
            else:
                cleaned_dict[key] = clean_json_recursive(value, target_keys)
        return cleaned_dict
    elif isinstance(data, list):
        return [clean_json_recursive(item, target_keys) for item in data]
    else:
        return data

# --- 主处理函数 (与之前版本相同) ---

def process_directory(input_dir: str, output_dir: str, target_keys: Set[str]):
    """
    处理输入目录中的所有 JSON 文件，清理指定字段（移除所有换行符），
    并将其保存到输出目录。
    """
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录未找到: {input_dir}")
        return

    if not target_keys:
        logger.warning("未指定要清理的目标键名。脚本将仅复制文件。")
        # return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"要清理（移除所有换行符）的目标键名: {', '.join(target_keys)}")

    processed_files = 0
    failed_files = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".json"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            logger.info(f"正在处理文件: {filename}")

            try:
                with open(input_filepath, "r", encoding="utf-8") as infile:
                    data = json.load(infile)

                cleaned_data = clean_json_recursive(data, target_keys)

                with open(output_filepath, "w", encoding="utf-8") as outfile:
                    json.dump(
                        cleaned_data,
                        outfile,
                        ensure_ascii=False,
                        indent=2,
                    )
                processed_files += 1
                logger.debug(f"成功处理并保存: {filename}")

            except json.JSONDecodeError as e:
                logger.error(f"无法解析文件 {filename} 中的 JSON: {e}")
                failed_files += 1
            except IOError as e:
                logger.error(f"处理文件 {filename} 时发生文件 I/O 错误: {e}")
                failed_files += 1
            except Exception as e:
                logger.error(f"处理文件 {filename} 时发生意外错误: {e}", exc_info=True)
                failed_files += 1
        else:
            logger.debug(f"跳过非 JSON 文件: {filename}")

    logger.info("--------------------------------------------------")
    logger.info("处理完成。")
    logger.info(f"成功处理的文件数: {processed_files}")
    logger.info(f"失败的文件数: {failed_files}")
    logger.info("--------------------------------------------------")


# --- 直接运行的入口点 (修改部分) ---

if __name__ == "__main__":
    # --- 在这里直接配置输入输出路径和目标字段 ---

    # 1. 指定包含原始 JSON 文件的输入目录路径
    #    使用 r"..." (原始字符串) 来避免 Windows 路径中反斜杠的问题
    input_directory = r"D:\GY\RAG_Standard\Parser\常用技术规范json_V2"

    # 2. 指定保存清理后 JSON 文件的新目录路径
    output_directory = r"D:\GY\RAG_Standard\Parser\常用技术规范json_V3"

    # 3. 指定要清理其换行符的字段名列表
    #    修改这个列表来指定你想要清理的字段
    target_keys_list = ["content"] # 例如，保持默认清理 "content"
    # target_keys_list = ["content", "summary", "description"] # 或者清理多个字段

    # --- 配置结束 ---

    # 将列表转换为集合，用于函数调用
    target_keys_set = set(target_keys_list)

    # 打印配置信息
    logger.info("脚本以直接运行模式启动。")
    logger.info(f"输入目录: {input_directory}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"目标字段: {target_keys_list}")

    # 调用主处理函数
    process_directory(input_directory, output_directory, target_keys_set)

    logger.info("脚本执行完毕。")