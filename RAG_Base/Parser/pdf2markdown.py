#使用docker部署的mineru来将pdf转为markdown格式的数据
import subprocess
import os
import shlex # 用于安全地处理命令参数

def run_mineru_docker(host_input_dir: str, host_output_dir: str, mineru_version: str = "1.2.2", timeout_seconds: int = 1800) -> tuple[bool, str]:
    """
    使用 Docker 运行 Mineru 工具进行 PDF 处理。

    参数:
    host_input_dir (str): 宿主机上包含输入文件的目录路径。
    host_output_dir (str): 宿主机上用于存放输出文件的目录路径。
    mineru_version (str): Mineru Docker 镜像的版本号。默认为 "1.2.2"。
    timeout_seconds (int): Docker 命令执行的超时时间（秒）。默认为 1800 秒 (30 分钟)。

    返回:
    tuple[bool, str]: 一个元组，第一个元素表示是否成功 (True/False)，
                      第二个元素是描述信息或错误信息。
    """
    # 0. 确保宿主机路径是绝对路径
    host_input_dir = os.path.abspath(host_input_dir)
    host_output_dir = os.path.abspath(host_output_dir)

    # 1. 确保宿主机输出目录存在
    try:
        os.makedirs(host_output_dir, exist_ok=True)
        print(f"信息: 确保宿主机输出目录存在: {host_output_dir}")
    except OSError as e:
        error_msg = f"错误: 创建输出目录 '{host_output_dir}' 失败: {e}"
        print(error_msg)
        return False, error_msg

    # 2. 定义 Docker 镜像和容器内路径
    docker_image = f"registry.cn-zhangjiakou.aliyuncs.com/6oclock/mineru:{mineru_version}"
    container_input_path = "/root/input"
    container_output_path = "/root/output"
    container_work_dir = container_input_path

    # 3. 构建在容器内执行的命令
    command_inside_container = (
        f"source /opt/mineru_venv/bin/activate && "
        f"magic-pdf -p {container_input_path} -o {container_output_path} -m auto"
    )

    # 4. 构建完整的 Docker 命令列表
    docker_command = [
        "docker", "run",
        # "--it", # 通常对于脚本化执行，-t (TTY) 不是必需的。如果需要stdin，-i可能就够了。
                 # 如果 magic-pdf 或其脚本确实需要 TTY，可能需要保留 -t，但这会使 subprocess 更难管理。
                 # 如果脚本完全非交互，可以尝试移除 -it。
        "--rm",
        "--gpus=all",
        "-v", f"{host_input_dir}:{container_input_path}",
        "-v", f"{host_output_dir}:{container_output_path}", # <--- 修正这里
        "-w", container_work_dir,
        docker_image,
        "/bin/bash", "-c", command_inside_container
    ]

    print(f"信息: 准备执行 Docker 命令: {' '.join(shlex.quote(arg) for arg in docker_command)}")

    try:
        # 5. 执行 Docker 命令
        process = subprocess.run(docker_command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
        
        success_msg = "Docker 命令执行成功!"
        if process.stdout:
            print("--- 标准输出 ---")
            print(process.stdout)
            success_msg += f"\n标准输出:\n{process.stdout.strip()}"
        if process.stderr: # Docker 常常将一些信息性内容输出到 stderr
            print("--- 标准错误 (可能包含有用信息) ---")
            print(process.stderr)
            # 有些工具即使成功也会在stderr输出信息，所以不一定代表错误
            # success_msg += f"\n标准错误输出:\n{process.stderr.strip()}" 
        
        return True, success_msg

    except subprocess.CalledProcessError as e:
        error_details = f"Docker 命令执行失败，返回码: {e.returncode}"
        if e.stdout:
            print("--- 标准输出 (错误时) ---")
            print(e.stdout)
            error_details += f"\n标准输出:\n{e.stdout.strip()}"
        if e.stderr:
            print("--- 标准错误 (错误时) ---")
            print(e.stderr)
            error_details += f"\n标准错误:\n{e.stderr.strip()}"
        print(f"错误: {error_details}")
        return False, error_details
    except subprocess.TimeoutExpired:
        error_msg = f"错误: Docker 命令执行超时 ({timeout_seconds}秒)!"
        print(error_msg)
        return False, error_msg
    except FileNotFoundError:
        error_msg = "错误: 'docker' 命令未找到。请确保 Docker 已安装并且在系统的 PATH 环境变量中。"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"错误: 执行 Docker 命令时发生未知错误: {e}"
        print(error_msg)
        return False, error_msg

# 这个 if __name__ == "__main__": 块主要用于直接测试此模块，
# 在被其他脚本导入时不会执行。
if __name__ == "__main__":
    print("正在直接运行 mineru_runner.py 进行测试...")
    
    # 示例：创建一个临时的输入目录和文件用于测试
    test_input_dir = os.path.abspath("./temp_mineru_test_input")
    test_output_dir = os.path.abspath("./temp_mineru_test_output")
    
    os.makedirs(test_input_dir, exist_ok=True)
    
    # 创建一个假的 PDF 文件（实际应用中你需要真实的 PDF）
    # 对于 magic-pdf，它可能期望真实的 PDF 文件。这里仅作结构演示。
    # 如果 magic-pdf 启动时找不到有效文件会报错，这是正常的。
    # 你需要将此路径替换为包含真实 PDF 的目录。
    # dummy_pdf_path = os.path.join(test_input_dir, "dummy.pdf")
    # with open(dummy_pdf_path, "w") as f:
    #     f.write("%PDF-1.4\n%%EOF") # 非常基础的PDF标记
    # print(f"创建了测试输入目录: {test_input_dir} (请确保里面有PDF文件)")

    # --- 配置你的路径 ---
    # !! 重要: 在实际测试时，请确保 test_input_dir 包含 magic-pdf 可以处理的文件 !!
    # 或者直接使用你已有的包含PDF文件的目录
    actual_test_input_dir = "/home/cdipd-admin/桌面/成规院政策文件" # 使用你实际的测试输入
    
    if not os.path.isdir(actual_test_input_dir):
         print(f"警告: 测试输入目录 '{actual_test_input_dir}' 不存在，跳过实际运行。")
         print(f"请修改 actual_test_input_dir 为一个包含PDF文件的有效目录，或在 {test_input_dir} 中放入PDF文件。")
    else:
        print(f"使用测试输入目录: {actual_test_input_dir}")
        print(f"测试输出将写入: {test_output_dir}")
        
        success, message = run_mineru_docker(actual_test_input_dir, test_output_dir, mineru_version="1.2.2")
        
        if success:
            print(f"\n测试运行成功: {message}")
            print(f"检查输出目录: {test_output_dir}")
        else:
            print(f"\n测试运行失败: {message}")

    # 清理临时测试目录 (可选)
    # import shutil
    # if os.path.exists(test_input_dir):
    #     shutil.rmtree(test_input_dir)
    # if os.path.exists(test_output_dir):
    #     shutil.rmtree(test_output_dir)
    # print("清理了临时测试目录。")