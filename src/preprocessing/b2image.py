import os
import math
import pefile
from PIL import Image
import csv
import argparse
from tqdm import tqdm


def calculate_shannon_entropy(data, window_size=32):
    """
    Calculates the Shannon entropy for a given data window.
    Returns a value between 0 and 8 (for byte data).
    """
    if not data:
        return 0.0

    entropy = 0.0
    # Calculate frequency of each byte value
    frequency = [0] * 256
    for byte in data:
        frequency[byte] += 1

    # Calculate entropy
    for freq in frequency:
        if freq > 0:
            probability = freq / window_size
            entropy -= probability * math.log2(probability)

    return entropy

def generate_pe_image_and_offset_map(file_path, output_image_path, output_map_path, image_width=256, entropy_window_size=32, silent=False):
    """
    Generates a color image representation of a PE binary and an offset map.

    Args:
        file_path (str): Path to the PE binary file.
        output_image_path (str): Path to save the generated image (e.g., "output.png").
        output_map_path (str): Path to save the offset map (e.g., "offset_map.txt").
        image_width (int): Desired width of the output image in pixels. Each row represents `image_width` bytes.
        entropy_window_size (int): Size of the sliding window for Shannon entropy calculation.
        silent (bool): If True, suppress print statements.
    """
    if not os.path.exists(file_path):
        if not silent:
            print(f"Error: File not found at {file_path}")
        return

    with open(file_path, 'rb') as f:
        binary_data = f.read()

    file_size = len(binary_data)
    if file_size == 0:
        if not silent:
            print(f"Error: File {file_path} is empty.")
        return

    # Calculate image dimensions
    image_height = math.ceil(file_size / image_width)
    total_pixels = image_width * image_height

    # Initialize RGB data for the image
    rgb_pixels = []
    offset_map_entries = []

    # --- PE Section G-channel mapping ---
    # This map assigns a G-channel value (0-255) to common PE sections.
    # You can customize these values based on your analytical needs.
    SECTION_G_MAP = {
        '.text': 240,       # Executable code
        '.code': 240,
        'CODE': 240,
        '.rdata': 200,      # Read-only data
        '.rodata': 200,
        '.data': 160,       # Initialized data
        '.idata': 120,      # Import table
        '.edata': 100,      # Export table
        '.rsrc': 80,        # Resources
        '.reloc': 60,       # Relocations
        '.pdata': 40,       # Exception handling
        '.bss': 20,         # Uninitialized data
        '.debug': 10,       # Debug information
    }
    # Default G value for unknown or non-section areas
    DEFAULT_G_VALUE = 0

    # --- Parse PE structure to get section information ---
    pe_sections_info = [] # List of (start_offset, end_offset, G_value, section_name)
    try:
        pe = pefile.PE(file_path, fast_load=True)
        for section in pe.sections:
            section_start = section.PointerToRawData
            section_end = section_start + section.SizeOfRawData
            section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
            section_name_lower = section_name.lower()
            g_value = SECTION_G_MAP.get(section_name_lower, DEFAULT_G_VALUE)
            pe_sections_info.append((section_start, section_end, g_value, section_name))
        pe.close()
    except pefile.PEFormatError:
        if not silent:
            print(f"Warning: {file_path} is not a valid PE file or corrupted. G-channel will be mostly default.")
    except Exception as e:
        if not silent:
            print(f"An unexpected error occurred during PE parsing: {e}")
    
    # Sort sections by start offset to easily determine current section
    pe_sections_info.sort(key=lambda x: x[0])
    current_section_idx = 0

    # --- Process each byte to generate RGB values and offset map ---
    for i in range(file_size):
        byte_value = binary_data[i]

        # R channel: Raw byte value
        r = byte_value

        # G channel: PE Section type
        g = DEFAULT_G_VALUE
        current_section_name = "PE_HEADER" if i < (pe_sections_info[0][0] if pe_sections_info else file_size) else "UNKNOWN"
        
        if pe_sections_info:
            while current_section_idx < len(pe_sections_info):
                sec_start, sec_end, sec_g_val, sec_name = pe_sections_info[current_section_idx]
                if sec_start <= i < sec_end:
                    g = sec_g_val
                    current_section_name = sec_name
                    break # Found current section
                elif i >= sec_end: # Past this section, move to next
                    current_section_idx += 1
                else: # i < sec_start, so it's a non-section area before current_section_idx
                    break
            
        # B channel: Shannon entropy
        # Determine the window for entropy calculation
        window_start = max(0, i - entropy_window_size // 2)
        window_end = min(file_size, i + entropy_window_size // 2 + (entropy_window_size % 2))
        
        # Adjust window if it goes past the end of the file
        if window_end - window_start < entropy_window_size:
            window_start = max(0, window_end - entropy_window_size)
            window_end = min(file_size, window_start + entropy_window_size) # Ensure window_end doesn't exceed file_size
            
        entropy_window = binary_data[window_start:window_end]
        
        # Calculate entropy and normalize to 0-255 (max entropy for bytes is 8)
        entropy = calculate_shannon_entropy(entropy_window, len(entropy_window))
        b = int((entropy / 8.0) * 255) # Scale 0-8 to 0-255

        rgb_pixels.append((r, g, b))

        # Store offset map entry: (x_coord, y_coord, file_offset, section_name)
        x = i % image_width
        y = i // image_width
        offset_map_entries.append(f"{x},{y},{i},{current_section_name}")

    # Pad if file size is not a multiple of image_width
    while len(rgb_pixels) < total_pixels:
        rgb_pixels.append((0, 0, 0)) # Pad with black pixels
        x = len(rgb_pixels) % image_width
        y = len(rgb_pixels) // image_width
        offset_map_entries.append(f"{x},{y},PAD,PAD") # Indicate padding in map

    # --- Create and save the image ---
    img = Image.new('RGB', (image_width, image_height))
    img.putdata(rgb_pixels)
    img.save(output_image_path)
    if not silent:
        print(f"Image saved to {output_image_path}")

    # --- Save the offset map ---
    with open(output_map_path, 'w') as f:
        f.write("pixel_x,pixel_y,file_offset,section\n")
        for entry in offset_map_entries:
            f.write(entry + "\n")
    if not silent:
        print(f"Offset map saved to {output_map_path}")



def batch_convert_from_csv(label_csv, extracted_root, output_root, image_width=256, entropy_window_size=32):
    """根据标签 CSV 对解压样本批量生成图像和偏移映射，带进度条和统计"""
    if not os.path.exists(label_csv):
        print(f"标签文件不存在: {label_csv}")
        return
    os.makedirs(output_root, exist_ok=True)
    total = 0
    success = 0
    with open(label_csv, newline='') as f:
        reader = list(csv.DictReader(f))
        # determine which column names to use (some CSVs use different headings)
        fieldnames = reader[0].keys() if reader else []
        sha_field = None
        fam_field = None
        for fn in fieldnames:
            lw = fn.lower()
            if 'sha' in lw:
                sha_field = fn
            if 'family' in lw:
                fam_field = fn
        if sha_field is None:
            raise ValueError(f"找不到 SHA256 字段，CSV 列名为: {fieldnames}")
        total = len(reader)
        for row in tqdm(reader, desc="转换进度", ncols=80):
            sha = row.get(sha_field, '').lower()
            family = row.get(fam_field, '').strip() if fam_field else ''
            in_path = os.path.join(extracted_root, family, sha + '.bin')
            if not os.path.exists(in_path):
                # 尝试无家族目录查找
                for fam in os.listdir(extracted_root):
                    candidate = os.path.join(extracted_root, fam, sha + '.bin')
                    if os.path.exists(candidate):
                        in_path = candidate
                        family = fam
                        break
            if not os.path.exists(in_path):
                tqdm.write(f"样本缺失: {sha} (family={family})")
                continue
            out_dir = os.path.join(output_root, family)
            os.makedirs(out_dir, exist_ok=True)
            img_path = os.path.join(out_dir, sha + '.png')
            map_path = os.path.join(out_dir, sha + '_map.txt')
            try:
                generate_pe_image_and_offset_map(in_path, img_path, map_path,
                                                image_width=image_width,
                                                entropy_window_size=entropy_window_size,
                                                silent=True)
                success += 1
            except Exception as e:
                tqdm.write(f"转换失败: {sha} ({e})")
    print(f"\n成功转换数量: {success} / {total}")

# --- Command-line interface for batch processing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from PE binaries based on label CSV")
    parser.add_argument("--labels", default="../new/data/Existing_labels.csv",
                        help="Path to label CSV file")
    parser.add_argument("--extracted", default="../new/data/malware_samples_pe",
                        help="Root directory of extracted PE samples")
    parser.add_argument("--outdir", default="../new/data/malware_images",
                        help="Output directory for generated images/maps")
    parser.add_argument("--width", type=int, default=256,
                        help="Image width in pixels")
    parser.add_argument("--entropy_window", type=int, default=32,
                        help="Entropy window size")
    parser.add_argument("--all", action="store_true", help="Ignore labels and convert every sample under extracted directory")
    parser.add_argument("--demo", action="store_true", help="Run demo example instead of batch")
    args = parser.parse_args()

    if not args.demo:
        if args.all:
            # build a temporary list of all samples under extracted dir
            # create pseudo-CSV entries
            temp_list = []
            for fam in os.listdir(args.extracted):
                fampath = os.path.join(args.extracted, fam)
                if not os.path.isdir(fampath):
                    continue
                for fname in os.listdir(fampath):
                    if fname.lower().endswith('.bin'):
                        sha = os.path.splitext(fname)[0].lower()
                        temp_list.append({'sha': sha, 'family': fam})
            # run conversion manually with progress using similar logic
            print(f"[*] 扫描到 {len(temp_list)} 个样本 (忽略标签)")
            total = len(temp_list)
            success = 0
            for row in tqdm(temp_list, desc="转换进度", ncols=80):
                sha = row['sha']
                family = row['family']
                in_path = os.path.join(args.extracted, family, sha + '.bin')
                if not os.path.exists(in_path):
                    tqdm.write(f"样本缺失: {sha} (family={family})")
                    continue
                out_dir = os.path.join(args.outdir, family)
                os.makedirs(out_dir, exist_ok=True)
                img_path = os.path.join(out_dir, sha + '.png')
                map_path = os.path.join(out_dir, sha + '_map.txt')
                try:
                    generate_pe_image_and_offset_map(in_path, img_path, map_path,
                                                    image_width=args.width,
                                                    entropy_window_size=args.entropy_window,
                                                    silent=True)
                    success += 1
                except Exception as e:
                    tqdm.write(f"转换失败: {sha} ({e})")
            print(f"\n成功转换数量: {success} / {total}")
            exit(0)
        else:
            batch_convert_from_csv(args.labels, args.extracted, args.outdir,
                                    image_width=args.width,
                                    entropy_window_size=args.entropy_window)
            # after batch conversion exit
            exit(0)

    # Demo fallback

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy PE-like file for testing (you would replace this with your actual .bin)
    # This is a very simplified example, real PE files are complex.
    # For a real test, use an actual executable file renamed to .bin
    dummy_pe_content = bytearray([
        0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, # MZ header part
        0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, # More DOS header
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, # e_lfanew = 0x80
        # ... fill up to 0x80 for PE Signature
    ])
    # Simulate a larger file for better image visualization
    # Extend with some 'code-like' bytes (low entropy) and 'data-like' bytes (mixed entropy)
    code_segment = bytearray([0x90, 0x90, 0xEB, 0xFE, 0x90, 0x90, 0x90, 0x90] * 50) # NOPs, low entropy
    data_segment = bytearray(os.urandom(200)) # Random bytes, high entropy
    resource_segment = bytearray([ord('H'), ord('e'), ord('l'), ord('l'), ord('o'), 0x00, 0x01, 0x02] * 30) # Mixed, lower entropy

    # A more complete dummy PE structure for testing pefile's section parsing
    # This is still highly simplified and might not be a 'valid' PE to Windows
    # but pefile should be able to parse sections if crafted correctly for offsets.
    # For robust testing, use an actual small executable file.
    # Let's create a file with recognizable sections for pefile
    # This requires more advanced pefile usage to construct a valid PE from scratch.
    # For the purpose of this example, assume you have a real PE file renamed to .bin
    
    # Use a real PE file (e.g., cmd.exe or a small legitimate executable) for better testing
    # Find a small executable on your system and rename it to 'test.bin' for this example
    # Example: copy C:\Windows\System32\cmd.exe test.bin
    
    # If you don't have a small PE file readily available for renaming,
    # you can just comment out the example usage and use your actual .bin file directly.

    # Example: If you have a file named 'malware.bin' in the same directory
    # replace 'dummy_pe.bin' with 'malware.bin'
    
    # For a robust test, it's best to use an actual PE file.
    # Let's create a very simple, albeit not fully compliant, PE structure for demonstration.
    # This will simulate sections for the G channel.
    
    # Let's create a more realistic dummy binary data
    # DOS Header (64 bytes)
    dos_header = bytearray([0x4D, 0x5A] + [0x00]*61 + [0x80, 0x00, 0x00, 0x00]) # e_lfanew at 0x3C points to 0x80
    
    # PE Header (Offset 0x80)
    pe_signature = bytearray([0x50, 0x45, 0x00, 0x00]) # "PE\0\0"
    
    # File Header (20 bytes following PE Signature)
    # Machine, NumberOfSections=2, TimeDateStamp, PointerToSymbolTable, NumberOfSymbols, SizeOfOptionalHeader, Characteristics
    file_header = bytearray([0x4C, 0x01, # Machine (i386)
                             0x02, 0x00, # NumberOfSections = 2
                             0x00, 0x00, 0x00, 0x00, # TimeDateStamp
                             0x00, 0x00, 0x00, 0x00, # PointerToSymbolTable
                             0x00, 0x00, 0x00, 0x00, # NumberOfSymbols
                             0xE0, 0x00, # SizeOfOptionalHeader (224 bytes for 32-bit PE)
                             0x02, 0x01])# Characteristics (EXECUTE|32BIT_MACHINE)

    # Optional Header (224 bytes) - Simplified for demo
    optional_header = bytearray([0x00]*224)
    
    # Section Headers (40 bytes each)
    # Section 1: .text (code)
    text_section_header = bytearray([
        0x2E, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, # .text name
        0x00, 0x10, 0x00, 0x00, # Virtual Size
        0x00, 0x10, 0x00, 0x00, # Virtual Address (e.g., 0x1000)
        0x00, 0x10, 0x00, 0x00, # SizeOfRawData (4KB)
        0x00, 0x04, 0x00, 0x00, # PointerToRawData (Offset 0x400)
        0x00, 0x00, 0x00, 0x00, # PointerToRelocations
        0x00, 0x00, 0x00, 0x00, # PointerToLinenumbers
        0x00, 0x00, # NumberOfRelocations
        0x00, 0x00, # NumberOfLinenumbers
        0x20, 0x00, 0x00, 0x60  # Characteristics (Executable, Readable)
    ])

    # Section 2: .data (data)
    data_section_header = bytearray([
        0x2E, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, # .data name
        0x00, 0x10, 0x00, 0x00, # Virtual Size
        0x00, 0x20, 0x00, 0x00, # Virtual Address (e.g., 0x2000)
        0x00, 0x10, 0x00, 0x00, # SizeOfRawData (4KB)
        0x00, 0x14, 0x00, 0x00, # PointerToRawData (Offset 0x1400) - needs to be after .text raw data
        0x00, 0x00, 0x00, 0x00, # PointerToRelocations
        0x00, 0x00, 0x00, 0x00, # PointerToLinenumbers
        0x00, 0x00, # NumberOfRelocations
        0x00, 0x00, # NumberOfLinenumbers
        0x40, 0x00, 0x00, 0xC0  # Characteristics (Writable, Readable)
    ])
    
    # Assemble dummy PE structure up to section headers
    # Ensure raw data offsets match section header PointersToRawData
    
    # Calculate sizes for proper offsets
    size_of_headers = len(dos_header) + len(pe_signature) + len(file_header) + len(optional_header) + \
                      len(text_section_header) + len(data_section_header) # Assuming 2 sections

    # Let's fix PointerToRawData and SizeOfRawData for simplicity
    # For real PE, these are aligned to FileAlignment
    text_raw_data_offset = 0x400 # Arbitrary, but after headers
    text_raw_data_size = 0x1000 # 4KB
    
    data_raw_data_offset = text_raw_data_offset + text_raw_data_size # Immediately after text
    data_raw_data_size = 0x800 # 2KB

    # Update section headers (very simplified)
    text_section_header[20:24] = text_raw_data_size.to_bytes(4, 'little') # SizeOfRawData
    text_section_header[24:28] = text_raw_data_offset.to_bytes(4, 'little') # PointerToRawData

    data_section_header[20:24] = data_raw_data_size.to_bytes(4, 'little') # SizeOfRawData
    data_section_header[24:28] = data_raw_data_offset.to_bytes(4, 'little') # PointerToRawData


    dummy_binary_content = dos_header + \
                           pe_signature + \
                           file_header + \
                           optional_header + \
                           text_section_header + \
                           data_section_header
    
    # Pad to text_raw_data_offset
    dummy_binary_content.extend([0x00] * (text_raw_data_offset - len(dummy_binary_content)))
    
    # Add raw section data
    dummy_binary_content.extend([0x90, 0xCC, 0xEB, 0xFE] * (text_raw_data_size // 4)) # Example code-like data
    dummy_binary_content.extend([0xAA, 0xBB, 0xCC, 0xDD] * (data_raw_data_size // 4)) # Example data-like data
    
    # Make sure the file is long enough for entropy window
    if len(dummy_binary_content) < 512:
        dummy_binary_content.extend([0x00] * (512 - len(dummy_binary_content)))


    dummy_file_path = "test_pe_binary.bin"
    with open(dummy_file_path, 'wb') as f:
        f.write(dummy_binary_content)
    
    # --- Run the generation ---
    print(f"Generating image and offset map for: {dummy_file_path}")
    generate_pe_image_and_offset_map(
        file_path=dummy_file_path,
        output_image_path="test_pe_binary_image.png",
        output_map_path="test_pe_binary_offset_map.txt"
    )

    # Clean up dummy file
    os.remove(dummy_file_path)
    print(f"Cleaned up {dummy_file_path}")
