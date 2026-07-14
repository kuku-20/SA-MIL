import os
import shutil



# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent

def reorganize_images_and_maps(root_dir):
    images_dir = os.path.join(root_dir, 'images')
    maps_dir = os.path.join(root_dir, 'maps')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    
    for family in os.listdir(root_dir):
        family_path = os.path.join(root_dir, family)
        if not os.path.isdir(family_path) or family in ['images', 'maps']:
            continue
        
        # Create subdirs in images and maps
        img_family_dir = os.path.join(images_dir, family)
        map_family_dir = os.path.join(maps_dir, family)
        os.makedirs(img_family_dir, exist_ok=True)
        os.makedirs(map_family_dir, exist_ok=True)
        
        for filename in os.listdir(family_path):
            src_path = os.path.join(family_path, filename)
            if filename.endswith('.png'):
                dst_path = os.path.join(img_family_dir, filename)
                shutil.move(src_path, dst_path)
                print(f"Moved image: {dst_path}")
            elif filename.endswith('_map.txt'):
                dst_path = os.path.join(map_family_dir, filename)
                shutil.move(src_path, dst_path)
                print(f"Moved map: {dst_path}")
        
        # Remove empty family dir
        if not os.listdir(family_path):
            os.rmdir(family_path)

if __name__ == '__main__':
    _PROJ_DIR / "data" / "malware_images'"
    reorganize_images_and_maps(root)
    print("Reorganization complete.")