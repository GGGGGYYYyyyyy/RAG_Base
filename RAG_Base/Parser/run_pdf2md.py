# 假设 mineru_runner.py 和这个脚本在同一目录
# 或者 mineru_runner.py 所在的目录在 PYTHONPATH 中
from pdf2markdown import run_mineru_docker
import os

def process_documents(input_folder: str, output_folder: str):
    """
    处理指定文件夹中的文档。
    """
    print(f"开始处理文档...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出将保存到: {output_folder}")

    # 调用 Docker 执行函数
    # 你可以根据需要传递不同的 mineru_version 或 timeout_seconds
    success, message = run_mineru_docker(
        host_input_dir=input_folder,
        host_output_dir=output_folder,
        mineru_version="1.2.2", # 可以按需更改
        timeout_seconds=3600    # 例如，设置1小时超时
    )

    if success:
        print(f"文档处理成功完成。")
        print(f"消息: {message}")
        # 这里可以添加后续操作，比如检查输出文件等
        # list_output_files(output_folder)
    else:
        print(f"文档处理失败。")
        print(f"错误信息: {message}")

def list_output_files(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"\n'{folder_path}' 中的文件列表:")
        for item in os.listdir(folder_path):
            print(os.path.join(folder_path, item))
    else:
        print(f"输出目录 '{folder_path}' 不存在或不是一个目录。")


if __name__ == "__main__":
    # --- 配置你的实际路径 ---
    # 这些路径是在运行 main_script.py 的服务器上的路径
    my_project_input_path = "/home/cdipd-admin/project_data/input_pdfs"
    my_project_output_path = "/home/cdipd-admin/project_data/processed_output"

    # 确保输入路径存在
    if not os.path.isdir(my_project_input_path):
        print(f"错误: 输入路径 '{my_project_input_path}' 不是一个有效的目录。请先创建并放入文件。")
        # 你也可以在这里尝试创建它，如果逻辑允许
        # os.makedirs(my_project_input_path, exist_ok=True)
        # print(f"已创建输入目录: {my_project_input_path}，请放入待处理文件。")
    else:
        process_documents(my_project_input_path, my_project_output_path)
        list_output_files(my_project_output_path) # 处理完后列出输出文件