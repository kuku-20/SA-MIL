
import numpy as np
import pefile
import os

def analyze_offset_and_section(offset_map_path, pe_file_path, y_min, y_max, width=256):
    """
    通过图像的 Y 坐标范围，在 Offset Map 中查找物理偏移，
    并自动映射到原始 PE 文件的具体 Section 中。
    """
    # 1. 加载 Offset Map (支持 .npy 和文本格式)
    if not os.path.exists(offset_map_path):
        print(f"❌ 找不到 Offset Map 文件: {offset_map_path}")
        return

    _, ext = os.path.splitext(offset_map_path)
    ext = ext.lower()
    offset_map = None
    try:
        if ext == '.npy':
            offset_map = np.load(offset_map_path)
        else:
            # 优先用 pandas 解析表格格式 (CSV/TXT 带列名)
            try:
                import pandas as pd
                df = pd.read_csv(offset_map_path, on_bad_lines='skip', engine='python')
                offset_map = df
            except Exception:
                # 退回到 numpy 文本加载（纯数值矩阵）
                try:
                    offset_map = np.loadtxt(offset_map_path)
                except Exception as e:
                    print(f"[WARN] 无法以 pandas 或 numpy 解析 offset_map: {e}")
                    offset_map = None
    except Exception as e:
        print(f"[WARN] 载入 offset_map 失败: {e}")
        offset_map = None
    
    # 2. 提取目标 Y 区域的物理偏移范围
    if offset_map is None:
        print("⚠️ 无法解析 offset_map 内容。请确认文件格式（.npy 或 文本表格/矩阵）。")
        return

    # 如果是 pandas DataFrame 或带 'pixel_y'/'offset' 表格
    try:
        import pandas as _pd
        is_df = isinstance(offset_map, _pd.DataFrame)
    except Exception:
        is_df = False

    if is_df:
        df = offset_map
        # 支持不同列名：'file_offset' 或 'offset'
        if 'pixel_y' in df.columns and ('file_offset' in df.columns or 'offset' in df.columns):
            off_col = 'file_offset' if 'file_offset' in df.columns else 'offset'
            valid_rows = df[(df['pixel_y'] >= y_min) & (df['pixel_y'] <= y_max)]
            valid_offsets = valid_rows[off_col].dropna().astype(int).values
        else:
            valid_offsets = np.array([])
    else:
        # 处理为 numpy 数组：支持 2D (H,W) 或 1D 展平
        arr = np.array(offset_map)
        if arr.ndim == 2:
            region = arr[y_min:y_max, :]
            valid_offsets = region[region != -1]
        else:
            start_idx = y_min * width
            end_idx = y_max * width
            region = arr[start_idx:end_idx]
            valid_offsets = region[region != -1]
        
    # 过滤掉填充值 (通常图像空白区域的 offset 会填充为 -1 或 0)
    # 0 在这里有可能是有效的 PE 头偏移，所以我们主要过滤掉 -1
    
    if len(valid_offsets) == 0:
        print("⚠️ 在该图像区域未找到有效的偏移量（可能全是 Padding）。")
        return
        
    min_offset = int(np.min(valid_offsets))
    max_offset = int(np.max(valid_offsets))
    
    print("="*60)
    print(f"📍 Traceback Step 2: Offset Mapping & Section Discovery")
    print("="*60)
    print(f"🖼️ Image Region       : Y = {y_min} to {y_max}")
    print(f"🔢 File Offset Range  : {min_offset} -> {max_offset}")
    print(f"⚡ Hex Format         : {hex(min_offset)} -> {hex(max_offset)}")
    print("-" * 60)

    # 3. 使用 pefile 自动映射到 PE Section
    if not os.path.exists(pe_file_path):
        print(f"❌ 找不到原始 PE 文件: {pe_file_path}")
        print("💡 你依然可以拿着上面打印的 Hex Offset 去 IDA 里手动定位。")
        return
        
    try:
        pe = pefile.PE(pe_file_path)
        print("🔍 正在逆向解析原始 PE 结构...")
        
        found_sections = []
        for section in pe.sections:
            sec_start = section.PointerToRawData
            sec_end = sec_start + section.SizeOfRawData
            sec_name = section.Name.decode('utf-8', errors='ignore').rstrip('\x00')
            
            # 检查偏移量是否有交集
            if max_offset >= sec_start and min_offset <= sec_end:
                overlap_start = max(min_offset, sec_start)
                overlap_end = min(max_offset, sec_end)
                found_sections.append({
                    'name': sec_name,
                    'start': hex(sec_start),
                    'end': hex(sec_end),
                    'overlap_bytes': overlap_end - overlap_start
                })
                
        if not found_sections:
            print("⚠️ 目标偏移没有落在任何标准的 Section 内。")
            # 检查是否在 PE 头
            if pe.sections and min_offset < pe.sections[0].PointerToRawData:
                print(f"🎯 结论: 该高光区域位于 【PE/DOS Header】 中！(Offset < {hex(pe.sections[0].PointerToRawData)})")
                print("   解释: 许多恶意软件（如 Gozi）会在 Header 的空隙中隐藏小型恶意 Payload 或配置。")
        else:
            for sec in found_sections:
                print(f"🎯 结论: 高光区域命中了 【{sec['name']}】 段！")
                print(f"   ├─ 该段物理范围: {sec['start']} - {sec['end']}")
                print(f"   └─ 命中的字节数: {sec['overlap_bytes']} bytes")
                
        print("="*60)
        print(f"🚀 下一步 (IDA Pro 分析指南):")
        print(f"1. 把原文件拖入 IDA Pro。")
        print(f"2. 按下键盘 'G' (Jump to address)。")
        print(f"3. 输入 {hex(min_offset)} 跳转过去。")
        print(f"4. 观察这里的汇编代码：是在解密(XOR)？还是异常的字符串？立刻截图！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ PE 文件解析失败: {e}")
        print("💡 请直接使用上方生成的 Hex Format 偏移量去 IDA 里查看。")

if __name__ == '__main__':
    # ================= 配置区 =================
    # 默认情况下尝试从上一步生成的临时 JSON 读取参数（.temp/trace_result.json）
    temp_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.temp', 'trace_result.json'))
    OFFSET_MAP_PATH = None
    PE_FILE_PATH = None
    Y_MIN = None
    Y_MAX = None

    if os.path.exists(temp_json):
        try:
            import json
            with open(temp_json, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            img_path = data.get('image_path')
            Y_MIN = int(data.get('y_min', 0))
            Y_MAX = int(data.get('y_max', Y_MIN + 224))
            # 尝试根据 images 路径推断 offset_map 的 .npy 路径
            if img_path and '/images/' in img_path:
                base_guess = img_path.replace('/images/', '/offset_maps/').rsplit('.png', 1)[0]
                # 优先尝试 .npy，其次尝试常见的文本 map 名称
                candidates = [base_guess + '.npy', base_guess + '_map.txt', base_guess + '_offset_map.txt', base_guess + '.txt']
                found = None
                for c in candidates:
                    if os.path.exists(c):
                        found = c
                        break
                if found:
                    OFFSET_MAP_PATH = found
                else:
                    OFFSET_MAP_PATH = candidates[0]
            print(f"[AUTO] 从上一步读取到: image={img_path}, y_min={Y_MIN}, y_max={Y_MAX}")
            print(f"[AUTO] 推断 offset_map 路径: {OFFSET_MAP_PATH}")
        except Exception as e:
            print(f"[WARN] 读取临时结果失败: {e}")

    # 如果仍未设置，请使用手动配置
    if OFFSET_MAP_PATH is None:
        OFFSET_MAP_PATH = "path/to/your/offset_map.npy"
    if PE_FILE_PATH is None:
        # 尝试根据 image 路径递归搜索原始 PE 文件（放在 new/data/malware_samples_pe 下的任意子目录）
        PE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'malware_samples_pe'))
        PE_FILE_PATH = "path/to/your/original_malware_file"
        try:
            if 'img_path' in locals() and img_path and '/images/' in img_path:
                basename = os.path.basename(img_path)
                stem = os.path.splitext(basename)[0]
                target_name = f"{stem}.bin"
                found_path = None
                for root, dirs, files in os.walk(PE_DIR):
                    if target_name in files:
                        found_path = os.path.join(root, target_name)
                        break
                if found_path:
                    PE_FILE_PATH = os.path.abspath(found_path)
                    print(f"[AUTO] 在 malware_samples_pe 中找到原始 PE: {PE_FILE_PATH}")
                else:
                    print(f"[AUTO] 未在 {PE_DIR} 下找到 {target_name}，请手动指定 PE_FILE_PATH")
        except Exception as e:
            print(f"[WARN] 推断 PE 文件时出错: {e}")
    if Y_MIN is None:
        Y_MIN = 0
    if Y_MAX is None:
        Y_MAX = 224

    analyze_offset_and_section(OFFSET_MAP_PATH, PE_FILE_PATH, Y_MIN, Y_MAX)
