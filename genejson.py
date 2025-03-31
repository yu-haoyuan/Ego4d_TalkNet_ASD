# -- get default json files --
# videoname_path is dataset/val/0d4efcc9-a336-46f9-a1db-0623b5be2869
# clip_path_f is dataset/val/0d4efcc9-a336-46f9-a1db-0623b5be2869/clip_f000000
# frames_path is dataset/val/0d4efcc9-a336-46f9-a1db-0623b5be2869/clip_f000000/frames
# -- get default json files --

# -- take ori-json data in newjson --
# in this module, video_clip is different from first module's clip
# first clip refers to 450 frames and second clip in json refers to clip-c241331....
# json_data is new json from ori-json
# -- take ori-json data in newjson --

import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_labels(split, mid_dir, base_dir):
    logging.info(f"Generating labels for {split} split")

    # 遍历视频目录
    for videoname_dir in os.listdir(mid_dir):
        videoname_path = os.path.join(mid_dir, videoname_dir)
        if not os.path.isdir(videoname_path):
            continue  # 确保是目录
        
        for clip_dir_f in os.listdir(videoname_path):
            clip_path_f = os.path.join(videoname_path, clip_dir_f)
            if not clip_dir_f.startswith("clip_f"):
                continue  # 只处理 clip_f 开头的文件夹
            
            json_name = clip_dir_f[-4:] + ".json"  # 取 `clip_fxxxxx` 后四位作为 JSON 文件名
            json_path = os.path.join(clip_path_f, json_name)

            # 重新处理所有 JSON，不再跳过已存在的 JSON 文件
            frames_path = os.path.join(clip_path_f, "frames")
            if os.path.exists(frames_path) and os.path.isdir(frames_path):
                frame_files = sorted(f for f in os.listdir(frames_path) if f.endswith(".jpg"))
                json_data = {str(int(frame.replace("img_", "").replace(".jpg", "").lstrip("0") or "0")): {} for frame in frame_files}

                # 保存 JSON 文件
                try:
                    with open(json_path, "w") as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    logging.error(f"Failed to write JSON {json_path}: {e}")

    # 读取原始 JSON 数据
    ori_json_path = os.path.join(base_dir, f"av_{split}.json")
    try:
        with open(ori_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Failed to read {ori_json_path}: {e}")
        return

    video_clip_num = 0
    for video in data.get("videos", []):
        for clip in video.get("clips", []):
            clip_uid = clip.get("clip_uid")
            video_clip_path = os.path.join(mid_dir, clip_uid)

            if not os.path.exists(video_clip_path):
                continue  # 目录不存在，跳过

            video_clip_num += 1
            logging.info(f"Processing video_clip {video_clip_num}: {clip_uid}")

            for person in clip.get("persons", []):
                person_id = person.get("person_id")
                camera_wear = person.get("camera_wearer", False)
                face_tracks = person.get("tracking_paths", []) if not camera_wear else 0
                voice_segments = person.get("voice_segments", [])

                # 遍历 clip 目录中的所有 JSON 文件
                for clip_path_f in [os.path.join(video_clip_path, f) for f in os.listdir(video_clip_path)]:
                    json_name = os.path.basename(clip_path_f)[-4:] + ".json"
                    json_path = os.path.join(clip_path_f, json_name)

                    if not os.path.exists(json_path):
                        continue  # 如果 JSON 不存在，则跳过

                    # **处理 JSON，即使它可能存在错误**
                    try:
                        with open(json_path, "r") as file:
                            json_data = json.load(file)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error reading {json_path}: {e}, regenerating...")
                        json_data = {}

                    # 遍历 JSON 文件中的帧数据
                    for frame_number in list(json_data.keys()):  # 避免动态修改 keys 导致错误
                        if not frame_number.isdigit():
                            continue  # 过滤掉非数字 key
                        frame_number_int = int(frame_number)

                        json_data[frame_number][person_id] = {
                            "camera_wearer": camera_wear,
                            "face": [] if face_tracks else 0,
                            "voice": 0
                        }

                        if face_tracks:
                            for track_id in face_tracks:
                                for track in track_id.get("track", []):
                                    if int(track["frame"]) == frame_number_int:
                                        json_data[frame_number][person_id]["face"].append({
                                            "x": track["x"],
                                            "y": track["y"],
                                            "width": track["width"],
                                            "height": track["height"]
                                        })
                        
                        if json_data[frame_number][person_id]["face"] == []:
                            json_data[frame_number][person_id]["face"] = 0

                        for voice in voice_segments:
                            if frame_number_int in range(voice["start_frame"], voice["end_frame"] + 1):
                                json_data[frame_number][person_id]["voice"] = 1
                                break

                    # **始终保存 JSON，即使原文件已存在**
                    try:
                        with open(json_path, "w") as file:
                            json.dump(json_data, file, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logging.error(f"Failed to update JSON {json_path}: {e}")

if __name__ == "__main__":
    base_dir = r"dataset"
    for split in ["val","train"]:
        mid_dir = os.path.join(base_dir, split)
        logging.info(f"Starting label generation for {split}")
        generate_labels(split, mid_dir, base_dir)
