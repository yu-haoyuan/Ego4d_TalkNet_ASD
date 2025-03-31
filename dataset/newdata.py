import os
import shutil
import math
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_BASE_DIR = Path("Ego4d_TalkNet_ASD/data")
OUTPUT_BASE_DIR = Path("Ego4d_TalkNet_ASD/dataset") # New output dir name
FRAMES_INPUT_DIR = INPUT_BASE_DIR / "video_imgs"
AUDIO_INPUT_DIR = INPUT_BASE_DIR / "wav"
SPLIT_DIR = INPUT_BASE_DIR / "split" # Directory containing train.list, val.list

FRAMES_PER_CLIP = 450
FPS = 30  # IMPORTANT: Assumed FPS. Change if necessary!
AUDIO_SAMPLE_RATE = 16000 # Target audio sample rate

# --- Helper Functions ---

def read_split_list(filepath):
    """Reads video IDs from a split list file."""
    if not filepath.exists():
        logging.error(f"Split file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
        logging.info(f"Read {len(video_ids)} IDs from {filepath}")
        return video_ids
    except Exception as e:
        logging.error(f"Error reading split file {filepath}: {e}")
        return []

def process_video(video_id, split, fps=30):
    """Processes a single video based on the new naming convention."""
    input_frames_dir = FRAMES_INPUT_DIR / video_id
    input_audio_path = AUDIO_INPUT_DIR / f"{video_id}.wav"
    output_video_dir = OUTPUT_BASE_DIR / split / video_id

    if not input_frames_dir.is_dir():
        logging.warning(f"Frame directory not found or not a directory for {video_id}, skipping.")
        return False
    if not input_audio_path.is_file():
        logging.warning(f"Audio file not found or not a file for {video_id}, skipping.")
        return False

    # --- Get total frame count (assuming frames are img_00001.jpg to img_NNNNN.jpg) ---
    try:
        frame_files = sorted(input_frames_dir.glob("img_*.jpg"))
        if not frame_files:
             logging.warning(f"No frames found in {input_frames_dir}, skipping.")
             return False
        # Infer frame count from the highest numbered frame file
        total_frames = int(frame_files[-1].stem.split('_')[-1]) # 1-based count
        logging.info(f"Processing {video_id}: Found {total_frames} frames (1-based).")
    except Exception as e:
        logging.error(f"Error counting frames for {video_id}: {e}")
        return False

    # --- Load Full Audio ---
    try:
        # logging.info(f"Loading audio: {input_audio_path}")
        full_audio = AudioSegment.from_wav(input_audio_path)
        # Optional: Resample and convert to mono if necessary
        if full_audio.frame_rate != AUDIO_SAMPLE_RATE:
             # logging.warning(f"Resampling audio for {video_id} from {full_audio.frame_rate} to {AUDIO_SAMPLE_RATE}")
             full_audio = full_audio.set_frame_rate(AUDIO_SAMPLE_RATE)
        if full_audio.channels != 1:
            # logging.warning(f"Converting audio for {video_id} to mono")
            full_audio = full_audio.set_channels(1)
    except Exception as e:
        logging.error(f"Error loading audio for {video_id}: {e}")
        return False # Cannot proceed without audio

    # --- Create Clips ---
    num_clips = math.ceil(total_frames / FRAMES_PER_CLIP)
    # logging.info(f"Splitting {video_id} into {num_clips} clips...")
    output_video_dir.mkdir(parents=True, exist_ok=True) # Create video-specific dir in output

    for clip_idx in range(num_clips):
        # Calculate frame range (0-based for internal calculations)
        start_frame_0based = clip_idx * FRAMES_PER_CLIP
        # End frame (exclusive for slicing, inclusive for naming)
        end_frame_0based_exclusive = min(start_frame_0based + FRAMES_PER_CLIP, total_frames)
        num_frames_in_this_clip = end_frame_0based_exclusive - start_frame_0based

        if num_frames_in_this_clip <= 0:
             continue # Should not happen with ceil, but safety check

        # Define directories and paths based on START frame
        clip_name = f"clip_f{start_frame_0based:06d}"
        output_clip_dir = output_video_dir / clip_name
        output_frames_dir = output_clip_dir / "frames"
        # Define audio path based on START and END (inclusive) frames
        end_frame_0based_inclusive = end_frame_0based_exclusive - 1
        output_audio_path = output_clip_dir / f"audio_f{start_frame_0based:06d}_f{end_frame_0based_inclusive:06d}.wav"

        output_frames_dir.mkdir(parents=True, exist_ok=True)

        # --- Copy Frames (Keeping Original Names) ---
        copied_frames_count = 0
        for i in range(num_frames_in_this_clip):
            current_frame_0based = start_frame_0based + i
            # Source filename uses 1-based indexing
            src_frame_num_1based = current_frame_0based + 1
            src_frame_name = f"img_{src_frame_num_1based:05d}.jpg"
            src_frame_path = input_frames_dir / src_frame_name

            # Destination filename is the SAME as source filename
            dst_frame_path = output_frames_dir / src_frame_name

            if src_frame_path.exists():
                try:
                    # Use copy instead of copyfile to preserve metadata if needed, though usually not necessary
                    shutil.copy(src_frame_path, dst_frame_path)
                    copied_frames_count += 1
                except Exception as e:
                    logging.warning(f"Could not copy frame {src_frame_path} to {dst_frame_path}: {e}")
            else:
                # This might happen if source frames are missing
                logging.warning(f"Source frame missing: {src_frame_path}")

        if copied_frames_count == 0 and num_frames_in_this_clip > 0 :
             logging.warning(f"  {clip_name}: No frames were copied for this clip. Skipping audio extraction.")
             # Optionally remove the empty clip directory
             # shutil.rmtree(output_clip_dir)
             continue

        # --- Extract Audio Segment ---
        # Calculate time in milliseconds based on 0-based frame indices
        start_time_ms = start_frame_0based * (1000 / fps)
        end_time_ms = end_frame_0based_exclusive * (1000 / fps) # pydub slicing is [start:end]

        try:
            audio_clip = full_audio[start_time_ms:end_time_ms]
            if len(audio_clip) > 0:
                 audio_clip.export(output_audio_path, format="wav")
            else:
                 logging.warning(f"  {clip_name}: Audio segment has zero length ({start_time_ms:.2f}ms to {end_time_ms:.2f}ms). Skipping audio export.")
                 # You might want to create a silent file of the expected duration
                 # expected_duration_ms = end_time_ms - start_time_ms
                 # if expected_duration_ms > 0:
                 #     silent_segment = AudioSegment.silent(duration=expected_duration_ms, frame_rate=AUDIO_SAMPLE_RATE)
                 #     silent_segment.export(output_audio_path, format="wav")


        except Exception as e:
            logging.error(f"Error extracting or saving audio for {clip_name} of {video_id}: {e}")

    # logging.info(f"Finished processing {video_id}")
    return True


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting dataset preparation with timestamp preservation...")

    # --- Create output directories ---
    train_dir = OUTPUT_BASE_DIR / "train"
    val_dir = OUTPUT_BASE_DIR / "val"
    # We create the base train/val dirs here. Video-specific dirs are created in process_video
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directories created/ensured: {train_dir}, {val_dir}")

    # --- Read Split Lists ---
    train_ids = read_split_list(SPLIT_DIR / "train.list")
    val_ids = read_split_list(SPLIT_DIR / "val.list") # Assuming val.list exists

    if not train_ids and not val_ids:
        logging.error("No video IDs loaded from split files. Please check paths and contents of .list files.")
        exit()
    if not train_ids:
        logging.warning("No training video IDs loaded.")
    if not val_ids:
        logging.warning("No validation video IDs loaded.")

    # --- Process Videos ---
    logging.info(f"\nProcessing {len(train_ids)} Training Videos...")
    train_success_count = 0
    for video_id in tqdm(train_ids, desc="Train Videos"):
        if process_video(video_id, "train", fps=FPS):
            train_success_count += 1
    logging.info(f"Successfully processed {train_success_count}/{len(train_ids)} training videos.")


    logging.info(f"\nProcessing {len(val_ids)} Validation Videos...")
    val_success_count = 0
    for video_id in tqdm(val_ids, desc="Val Videos"):
         if process_video(video_id, "val", fps=FPS):
             val_success_count += 1
    logging.info(f"Successfully processed {val_success_count}/{len(val_ids)} validation videos.")


    logging.info("Dataset preparation finished.")