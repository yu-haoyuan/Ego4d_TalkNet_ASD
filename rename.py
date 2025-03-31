import os
import re

def find_frames_dirs(base_path):
    frames_dirs = []
    for root, dirs, _ in os.walk(base_path):
        if root.endswith("frames"):
            frames_dirs.append(root)
    return frames_dirs

def rename_images(base_path):
    pattern = re.compile(r"img_(\d{5})\.jpg")  # 匹配 img_00001.jpg 形式的文件
    base_dirs = find_frames_dirs(base_path)
    
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            # 先筛选出符合条件的文件，并按数字排序（正序处理）
            image_files = sorted([f for f in files if pattern.match(f)])
            temp_names = {}
            
            # 第一步：先改成临时名字，避免覆盖
            for file in image_files:
                match = pattern.match(file)
                if match:
                    old_number = int(match.group(1))
                    new_number = old_number - 1
                    temp_filename = f"temp_{new_number:05d}.jpg"
                    old_path = os.path.join(root, file)
                    temp_path = os.path.join(root, temp_filename)
                    
                    os.rename(old_path, temp_path)
                    temp_names[temp_filename] = new_number  # 记录新编号
                    print(f"Temporarily renamed: {old_path} -> {temp_path}")
            
            # 第二步：再改回最终名字
            for temp_filename, new_number in temp_names.items():
                final_filename = f"img_{new_number:05d}.jpg"
                temp_path = os.path.join(root, temp_filename)
                final_path = os.path.join(root, final_filename)
                
                os.rename(temp_path, final_path)
                print(f"Final rename: {temp_path} -> {final_path}")

if __name__ == "__main__":
    base_path = "dataset"
    rename_images(base_path)
