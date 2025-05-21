import os
import argparse
import json
from EnhancedMarkdownParser import EnhancedMarkdownParser 
import sys
# --- 添加项目根目录到 sys.path ---
# 获取当前脚本 (batch_process_markdown.py) 所在的目录 (Parser/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 Parser/ 目录的父目录，即项目根目录 (RAG_STANDARD_V2/)
project_root_dir = os.path.dirname(current_script_dir)
# 将项目根目录添加到 sys.path 的开头，使其具有高优先级
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- 完成路径添加 ---

# 现在可以从 config 包导入模块了
# 假设您在 RAG_STANDARD_V2/config/ 文件夹下有一个名为 settings.py 或 db_settings.py 的文件
# 如果您的配置文件就叫 RAG_STANDARD_V2/config/config.py，则导入语句是：
try:
    from config import config  # 将导入的模块重命名为 app_config 以免与局部变量冲突
    # 或者如果您的配置文件是 RAG_STANDARD_V2/config/settings.py:
    # from config import settings as app_config
    print("成功从 config 包导入配置。")
except ModuleNotFoundError:
    print(f"错误：无法从 '{project_root_dir}/config' 导入配置模块。请确保该路径下有配置文件 (例如 config.py 或 settings.py) 并且 config 目录是一个有效的 Python 包 (通常意味着它可能需要一个空的 __init__.py 文件，尽管对于简单的模块导入有时不是严格必需的)。")
    sys.exit(1) # 导入失败则退出
except ImportError as e:
    print(f"导入配置时出错: {e}")
    sys.exit(1)
def load_db_mapping(mapping_file_path: str) -> dict:
    """从JSON文件加载文件名到数据库resourceName的映射"""
    if not mapping_file_path or not os.path.exists(mapping_file_path):
        return {}
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"成功从 '{mapping_file_path}' 加载了 {len(mapping)} 条数据库映射。")
        return mapping
    except Exception as e:
        print(f"警告：加载数据库映射文件 '{mapping_file_path}' 失败: {e}")
        return {}

def process_markdown_folder(
    input_folder: str,
    output_folder: str = None,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
    contain_closest_title_levels: int = 2,
    convert_table_to_text: bool = True,
    remove_hyperlinks: bool = False,
    remove_images: bool = False,
    encoding: str = "utf-8",
    # db_config 现在会从 config.py 构建
    target_db_extension: str = ".pdf",
    db_mapping_file: str = None,
    use_database: bool = True # 新增: 控制是否尝试使用数据库
):
    """
    批量处理文件夹下的所有Markdown文件，并将结果保存为JSON文件
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'json_output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    db_resource_mapping = load_db_mapping(db_mapping_file)
    
    db_connection_params = None
    if use_database:
        if config.DB_HOST and config.DB_USER and config.DB_NAME:
            db_connection_params = {
                'host': config.DB_HOST,
                'port': config.DB_PORT,
                'user': config.DB_USER,
                'password': config.DB_PASSWORD,
                'database': config.DB_NAME,
                'charset': config.DB_CHARSET
            }
            print(f"将使用配置文件中的数据库设置: host={config.DB_HOST}, port={config.DB_PORT}, user={config.DB_USER}, db={config.DB_NAME}")
        else:
            print("警告：配置文件中未提供完整的数据库主机、用户或名称。将不连接数据库。")
            use_database = False # 强制不使用数据库
    else:
        print("配置为不使用数据库。")

    parser_instance = EnhancedMarkdownParser(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        contain_closest_title_levels=contain_closest_title_levels,
        convert_table_to_text=convert_table_to_text,
        remove_hyperlinks=remove_hyperlinks,
        remove_images=remove_images,
        encoding=encoding,
        db_config=db_connection_params if use_database else None, # 传递数据库配置
        target_db_extension=target_db_extension
    )
    
    md_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.md', '.markdown')):
                md_files.append(os.path.join(root, file))
    
    total_files = len(md_files)
    print(f"找到 {total_files} 个Markdown文件进行处理。")
    
    processed_count = 0
    failed_count = 0

    for i, md_file_path in enumerate(md_files, 1):
        try:
            file_basename = os.path.basename(md_file_path)
            doc_title_for_parser = os.path.splitext(file_basename)[0] 

            db_lookup_key = None
            if use_database and file_basename in db_resource_mapping:
                db_lookup_key = db_resource_mapping[file_basename]
                print(f"    为 '{file_basename}' 使用映射的数据库键: '{db_lookup_key}'")
            
            rel_path = os.path.relpath(md_file_path, input_folder)
            output_file_dir = os.path.dirname(os.path.join(output_folder, rel_path))
            os.makedirs(output_file_dir, exist_ok=True)
            
            output_json_filename = os.path.join(output_file_dir, f"{doc_title_for_parser}.json")
            
            print(f"[{i}/{total_files}] 处理文件: {rel_path}")
            
            chunks = parser_instance.parse_file(
                file_path=md_file_path,
                doc_title=doc_title_for_parser,
                remove_hyperlinks=remove_hyperlinks,
                remove_images=remove_images,
                db_resource_name_for_metadata=db_lookup_key if use_database else None
            )
            
            if chunks:
                with open(output_json_filename, 'w', encoding=encoding) as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"    ✓ 成功生成 {len(chunks)} 个块，保存到 {output_json_filename}")
                processed_count +=1
            else:
                print(f"    ✗ 文件解析结果为空或未生成块: {md_file_path}")
                
        except Exception as e:
            print(f"    ✗ 处理文件 '{md_file_path}' 时发生严重错误: {str(e)}")
            failed_count += 1
    
    if use_database: # 仅当尝试使用数据库时才关闭连接
        parser_instance.close_db_connection()

    print(f"\n批处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理文件数: {processed_count}")
    print(f"处理失败或未生成块的文件数: {total_files - processed_count}")
    print(f"结果保存在: {output_folder}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="批量处理Markdown文件并转换为JSON，可选数据库元数据集成 (数据库配置从config.py加载)")
    cli_parser.add_argument("input_folder", help="输入Markdown文件夹路径")
    cli_parser.add_argument("--output_folder", "-o", help="输出JSON文件夹路径 (默认: input_folder/json_output)")
    
    cli_parser.add_argument("--max_chunk_size", type=int, default=1000, help="文本块的最大大小 (默认: 1000)")
    cli_parser.add_argument("--chunk_overlap", type=int, default=200, help="块之间的重叠大小 (默认: 200)")
    cli_parser.add_argument("--title_levels", type=int, default=2, help="包含的最近标题层级数量 (默认: 2)")
    cli_parser.add_argument("--no_table_convert", action="store_false", dest="convert_table_to_text", help="不将表格转换为文本描述 (默认: 转换)")
    cli_parser.add_argument("--remove_links", action="store_true", help="移除超链接 (默认: 不移除)")
    cli_parser.add_argument("--remove_images", action="store_true", help="移除图片 (默认: 不移除)")
    cli_parser.add_argument("--encoding", default="utf-8", help="文件编码 (默认: utf-8)")

    # 数据库使用控制及相关文件参数
    cli_parser.add_argument("--no_db", action="store_false", dest="use_database", help="不尝试连接或使用数据库进行元数据查找 (默认: 使用数据库，需config.py配置正确)")
    cli_parser.add_argument("--target_db_extension", default=".pdf", help="用于从文件名推断数据库键的目标扩展名 (默认: .pdf), 当未使用映射文件且使用数据库时生效")
    cli_parser.add_argument("--db_mapping_file", help="JSON文件路径，包含从输入文件名到数据库resourceName的映射。例如: {\"input.md\": \"db_key.pdf\"}")

    args = cli_parser.parse_args()
    
    # 数据库参数现在从 config.py 读取，不再通过命令行传递给 process_markdown_folder
    # use_database 控制是否实际使用这些配置

    process_markdown_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap,
        contain_closest_title_levels=args.title_levels,
        convert_table_to_text=args.convert_table_to_text,
        remove_hyperlinks=args.remove_links,
        remove_images=args.remove_images,
        encoding=args.encoding,
        target_db_extension=args.target_db_extension,
        db_mapping_file=args.db_mapping_file,
        use_database=args.use_database # 传递是否使用数据库的标志
    )

# python /home/cdipd-admin/RAG_Standard_V2/Parser/batch_process_markdown.py \
#   /home/cdipd-admin/RAG_Standard_V2/data/markdown \
#   --output_folder /home/cdipd-admin/RAG_Standard_V2/data/json \
#   --max_chunk_size 512 \
#   --chunk_overlap 128 \
#   --remove_links \
#   --remove_images \
#   --db_mapping_file /path/to/your/mapping_details.json 