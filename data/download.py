
import os
import time
import requests
import pandas as pd
from requests.exceptions import RequestException
from tqdm import tqdm

# ==========================================
# 核心配置区域 (Configuration)
# ==========================================
API_KEY = "e488b137918310f1524f011d74216d7cd54277a6654580b1"
API_URL = "https://mb-api.abuse.ch/api/v1/"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "labels", "MalwareBazaar_Labels.csv")
BASE_DOWNLOAD_DIR = os.path.join(_SCRIPT_DIR, "malware_samples_raw")

# 接口限制：为了防止被 MalwareBazaar 封禁 IP，建议在每次请求之间加上适当延迟
# 官方说明有日下载量限制，如果数据量大（如近 4000 个），建议分批下载或设置延迟
SLEEP_TIME = 0.5 

def download_malware_samples():
    print(f"[*] 正在初始化下载环境，存储目录：{BASE_DOWNLOAD_DIR}")
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)
    
    # 1. 加载 CSV 标签文件
    try:
        df = pd.read_csv(CSV_PATH)
        total_samples = len(df)
        print(f"[*] 成功加载标签文件，共发现 {total_samples} 个样本记录。")
    except Exception as e:
        print(f"[!] 无法读取 CSV 文件: {e}")
        return

    headers = {
        "Auth-Key": API_KEY
    }

    success_count = 0
    fail_count = 0
    skip_count = 0

    # 2. 遍历每一行数据
    for index, row in tqdm(df.iterrows(), total=len(df), desc="下载进度"):
        sha256 = str(row['Malware SHA-256']).strip()
        family = str(row['Family']).strip()
        
        # 根据家族名称创建子文件夹，例如：malware_samples_raw/Gozi/
        family_dir = os.path.join(BASE_DOWNLOAD_DIR, family)
        os.makedirs(family_dir, exist_ok=True)
        
        # 定义输出文件路径，统一保存为 .zip 格式
        out_filepath = os.path.join(family_dir, f"{sha256}.zip")
        
        # 3. 断点续传机制：检查文件是否已存在
        if os.path.exists(out_filepath):
            skip_count += 1
            continue
            
        data = {
            "query": "get_file",
            "sha256_hash": sha256
        }

        # 4. 发起 API 请求下载
        try:
            response = requests.post(API_URL, headers=headers, data=data, timeout=30)
            
            # 5. 校验响应内容
            # 成功下载的恶意软件 ZIP 文件，其二进制内容开头必定是 'PK' (标准 ZIP 文件头)
            if response.status_code == 200 and response.content.startswith(b'PK'):
                with open(out_filepath, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                fail_count += 1
                # 有些样本可能已经被平台下架 (file_not_found)
                
        except RequestException as e:
            fail_count += 1
            
        # 6. API 速率限制保护
        time.sleep(SLEEP_TIME)

    print("\n==========================================")
    print("下载任务总结:")
    print(f"  总计划数: {total_samples}")
    print(f"  成功下载: {success_count}")
    print(f"  跳过(已存在): {skip_count}")
    print(f"  失败/未找到: {fail_count}")
    print(f"  所有样本默认密码: infected")
    print("==========================================")

if __name__ == "__main__":
    download_malware_samples()
