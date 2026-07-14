import pandas as pd
import pefile
import os
import json
import sys



# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

def analyze_offset_csv(offset_map_path, pe_file_path, y_min, y_max):
    """
    通过图像的 Y 坐标范围，在 txt/csv 格式的 Offset Map 中查找物理偏移，
    并自动映射到原始 PE 文件的具体 Section 中。
    """
    if not os.path.exists(offset_map_path):
        print(f"❌ 找不到 Offset Map 文件: {offset_map_path}")
        return False
        
    print(f"📄 正在读取 Offset Map: {os.path.basename(offset_map_path)}...")
    
    try:
        # 读取 txt 文件 (CSV格式)，指定引擎加速
        df = pd.read_csv(
            offset_map_path,
            usecols=['pixel_y', 'file_offset', 'section'],
            dtype={'pixel_y': int, 'section': str},
            on_bad_lines='skip',
            engine='c'
        )
    except Exception as e:
        print(f"❌ 读取 Offset Map 失败: {e}")
        return False
    
    # 过滤出落在目标 Y 区域的像素
    target_region = df[(df['pixel_y'] >= y_min) & (df['pixel_y'] <= y_max)].copy()
    
    # 过滤掉 PAD (填充像素)
    target_region = target_region[target_region['file_offset'] != 'PAD']
    
    if target_region.empty:
        print("⚠️ 在该图像区域未找到有效的偏移量（全是 Padding）。")
        return False
        
    # 转换类型并获取最小/最大偏移
    try:
        offsets = pd.to_numeric(target_region['file_offset'], errors='coerce')
        offsets = offsets.dropna()
    except:
        print("❌ 无法解析偏移量数据")
        return False
    
    if len(offsets) == 0:
        print("⚠️ 没有有效的数值偏移量")
        return False
    
    min_offset = int(offsets.min())
    max_offset = int(offsets.max())
    
    print("="*60)
    print(f"📍 Traceback Step 2: Offset Mapping & Section Discovery")
    print("="*60)
    print(f"🖼️ Image Region       : Y = {y_min} to {y_max}")
    print(f"📊 匹配像素数         : {len(target_region)}")
    print(f"🔢 File Offset Range  : {min_offset} -> {max_offset}")
    print(f"⚡ Hex Format         : {hex(min_offset)} -> {hex(max_offset)}")
    
    # 统计该区域内主要属于哪个 Section
    section_counts = target_region['section'].value_counts()
    print("-" * 60)
    print("📊 区域内的 Section 占比:")
    for sec, count in section_counts.items():
        pct = (count / len(target_region)) * 100
        print(f"   - {sec}: {count} 像素 ({pct:.1f}%)")
    print("-" * 60)

    # 3. 使用 pefile 解析
    if not os.path.exists(pe_file_path):
        print(f"⚠️ 找不到原始 PE 文件: {pe_file_path}")
        print("💡 你可以直接去 IDA 跳转到上面的 Hex Offset。")
        return True
        
    try:
        pe = pefile.PE(pe_file_path)
        print("🔍 正在通过 pefile 验证原始 PE 结构...")
        
        found = False
        for section in pe.sections:
            sec_start = section.PointerToRawData
            sec_end = sec_start + section.SizeOfRawData
            sec_name = section.Name.decode('utf-8', errors='ignore').rstrip('\x00')
            
            # 检查偏移范围是否与该 Section 重叠
            if not (max_offset < sec_start or min_offset > sec_end):
                overlap_start = max(min_offset, sec_start)
                overlap_end = min(max_offset, sec_end)
                overlap_size = overlap_end - overlap_start
                print(f"🎯 物理匹配: 【{sec_name}】 段 (区间: {hex(sec_start)} - {hex(sec_end)})")
                print(f"   └─ 重叠范围: {hex(overlap_start)} - {hex(overlap_end)} ({overlap_size} bytes)")
                found = True
                
        if not found:
            dos_header_size = pe.sections[0].PointerToRawData if pe.sections else 0x1000
            if min_offset < dos_header_size:
                print(f"🎯 物理匹配: 【PE / DOS Header】 区间 (< {hex(dos_header_size)})")
                found = True
        
        if not found:
            print(f"⚠️ 未能精确匹配到 PE 中的 Section，但偏移有效。")
            
        print("="*60)
        print(f"🚀 下一步 (IDA Pro 分析):")
        print(f"1. 把原文件拖入 IDA Pro。")
        print(f"2. 键盘按 'G', 跳转到: {hex(min_offset)}")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"⚠️ PE 文件解析失败: {e}")
        print("💡 但偏移信息仍然有效，可在 IDA 中手动跳转。")
        return True

def resolve_map_path(image_path):
    """根据图片路径自动查找对应的 offset map"""
    # 示例: /path/to/images/Gozi/8e5d...png -> /path/to/offset_maps/Gozi/8e5d..._map.txt
    img_dir = os.path.dirname(image_path)
    img_name = os.path.basename(image_path).replace('.png', '')
    
    # 替换 images -> offset_maps
    map_dir = img_dir.replace('/images/', '/offset_maps/')
    
    candidates = [
        os.path.join(map_dir, f"{img_name}_map.txt"),
        os.path.join(map_dir, f"{img_name}_offset_map.txt"),
    ]
    
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def resolve_pe_path(image_path):
    """根据图片路径自动查找对应的 PE 文件"""
    # 示例: /path/to/images/Gozi/8e5d...png -> /path/to/malware_samples_pe/Gozi/8e5d....bin
    img_dir = os.path.dirname(image_path)
    img_name = os.path.basename(image_path).replace('.png', '')
    family = os.path.basename(img_dir)
    
    pe_dir = img_dir.replace('/images/', '/malware_samples_pe/')
    
    candidates = [
        os.path.join(pe_dir, f"{img_name}.bin"),
        os.path.join(pe_dir, f"{img_name}"),
    ]
    
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

if __name__ == '__main__':
    # ================= 配置区 =================
    
    # 方案 A: 手动指定（支持旧版用法）
    USE_AUTO_RESOLVE = True  # 改为 False 使用手动配置
    
    if not USE_AUTO_RESOLVE:
OFFSET_MAP_PATH = str(_PROJ_DIR / "data" / "malware_images" / "offset_maps" / "Gozi" / "8e5d52727fd76e7fc3078c8bc3607e8d0fc2b4d9eaf09de824c59f2ed26b0f21_map.txt")
PE_FILE_PATH = str(_PROJ_DIR / "data" / "malware_samples_pe" / "Gozi" / "8e5d52727fd76e7fc3078c8bc3607e8d0fc2b4d9eaf09de824c59f2ed26b0f21.bin")
        Y_MIN = 672
        Y_MAX = 896
    else:
        # 方案 B: 从 trace_attention.py 的输出读取
        temp_result = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.temp', 'trace_result.json'))
        
        if os.path.exists(temp_result):
            print(f"📖 从 {temp_result} 读取 trace 结果...")
            with open(temp_result, 'r', encoding='utf-8') as fh:
                result = json.load(fh)
            image_path = result['image_path']
            Y_MIN = result['y_min']
            Y_MAX = result['y_max']
            
            OFFSET_MAP_PATH = resolve_map_path(image_path)
            PE_FILE_PATH = resolve_pe_path(image_path)
            
            if not OFFSET_MAP_PATH:
                print(f"❌ 无法自动解析 Offset Map，请手动设置")
                sys.exit(1)
            if not PE_FILE_PATH:
                print(f"⚠️ 无法自动解析 PE 文件路径")
                PE_FILE_PATH = ""
        else:
            print(f"❌ 找不到 trace 结果: {temp_result}")
            print("💡 请先运行: python trace_attention.py")
            sys.exit(1)
    
    # ==========================================
    
    analyze_offset_csv(OFFSET_MAP_PATH, PE_FILE_PATH, Y_MIN, Y_MAX)
