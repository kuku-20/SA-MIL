
import os
import zipfile
import shutil
import subprocess
import tempfile

# ==========================================
# 核心配置区域
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_SCRIPT_DIR, "malware_samples_raw")
UNPACKED_DIR = os.path.join(_SCRIPT_DIR, "malware_samples_pe")
ZIP_PASSWORD = b"infected"            # zipfile 用的 byte 密码
ZIP_PASSWORD_STR = "infected"         # 命令行工具用的 string 密码

def is_pe_file(filepath):
    """通过检查 MZ 头部判断是否为有效 PE 文件"""
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'MZ'
    except Exception:
        return False

def extract_archive(zip_path, temp_dir):
    """
    尝试多种解压工具将文件提取到指定的临时目录中。
    按成功率和兼容性排序：1. 7z (支持AES) -> 2. unzip -> 3. zipfile
    """
    # 1. 优先尝试 7z (处理 MalwareBazaar 的 AES 密码压缩包最靠谱)
    seven = shutil.which('7z') or shutil.which('7zr')
    if seven:
        try:
            subprocess.run([seven, 'x', f'-p{ZIP_PASSWORD_STR}', '-y', f'-o{temp_dir}', zip_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass

    # 2. 尝试 Mac/Linux 自带的 unzip
    unzip = shutil.which('unzip')
    if unzip:
        try:
            subprocess.run(['unzip', '-P', ZIP_PASSWORD_STR, '-j', '-d', temp_dir, zip_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass

    # 3. 尝试 Python 原生 zipfile (兜底，但不支持 AES 加密)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path=temp_dir, pwd=ZIP_PASSWORD)
            return True
    except Exception:
        pass

    return False

def unpack_and_filter():
    print(f"[*] 准备批量解压，源目录: {RAW_DIR}")
    print(f"[*] 目标输出目录: {UNPACKED_DIR}")
    
    if not os.path.exists(RAW_DIR):
        print(f"[!] 找不到源目录 {RAW_DIR}")
        return

    os.makedirs(UNPACKED_DIR, exist_ok=True)
    
    total_zips = 0
    extracted_pe_count = 0
    invalid_pe_count = 0
    error_count = 0

    for family_name in os.listdir(RAW_DIR):
        family_dir = os.path.join(RAW_DIR, family_name)
        if not os.path.isdir(family_dir) or family_name.startswith('.'):
            continue
            
        out_family_dir = os.path.join(UNPACKED_DIR, family_name)
        os.makedirs(out_family_dir, exist_ok=True)
        
        for filename in os.listdir(family_dir):
            if not filename.endswith('.zip'):
                continue
                
            total_zips += 1
            zip_path = os.path.join(family_dir, filename)
            sha256 = filename.replace('.zip', '')
            final_pe_path = os.path.join(out_family_dir, f"{sha256}.bin")
            
            # 断点续传
            if os.path.exists(final_pe_path):
                print(f"[-] [{family_name}] 跳过已解压文件: {sha256[:10]}...")
                extracted_pe_count += 1
                continue
                
            print(f"[*] 正在处理 [{family_name}] : {sha256[:10]}...")
            
            # 使用 tempfile 创建用完即毁的临时沙盒目录
            with tempfile.TemporaryDirectory() as temp_dir:
                success = extract_archive(zip_path, temp_dir)
                
                if not success:
                    print("    [!] 所有解压尝试均失败 (密码错误或压缩包损坏)")
                    error_count += 1
                    continue
                
                # 获取临时沙盒中解压出来的文件
                extracted_files = os.listdir(temp_dir)
                if not extracted_files:
                    print("    [!] 解压成功，但里面没有文件")
                    error_count += 1
                    continue
                
                # MalwareBazaar 的压缩包通常只有一个文件
                target_file_path = os.path.join(temp_dir, extracted_files[0])
                
                # 严格校验：是否为有效 PE 文件
                if os.path.isfile(target_file_path) and is_pe_file(target_file_path):
                    # 安全转移并重命名为 .bin
                    shutil.move(target_file_path, final_pe_path)
                    extracted_pe_count += 1
                    print("    [+] 成功提取并验证为合法 PE 文件！")
                else:
                    invalid_pe_count += 1
                    print("    [!] 丢弃：非 PE 文件 (无 MZ 头) 或内含文件夹。")

    print("\n==========================================")
    print("安全解压与 PE 提取总结:")
    print(f"  共扫描 ZIP: {total_zips}")
    print(f"  提取有效 PE: {extracted_pe_count}")
    print(f"  过滤无效文件: {invalid_pe_count}")
    print(f"  解压出错: {error_count}")
    print("==========================================")

if __name__ == "__main__":
    unpack_and_filter()
