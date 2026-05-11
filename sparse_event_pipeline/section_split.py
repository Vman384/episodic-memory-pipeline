import os
import subprocess
import argparse
import math
import json
from tqdm import tqdm

def get_vid_duration(vid_path: str):
     """
     Gets the duration of video using ffprobe
     """
     command = [
          'ffprobe',
          '-v', 'error',
          '-show_entries', 'format=duration',
          '-of', 'default=noprint_wrappers=1:nokey=1',
          vid_path
     ]
     
     try:
          # run the subprocess commands listed above
          result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
          return float(result.stdout.strip())
     
     except subprocess.CalledProcessError as e:
          print(f"Error getting duration for {vid_path}: {e.stderr}")

def split_video(vid_path:str, output: str, section_duration=300, overlap=10):
     """
     Splits video into overlapping sections using ffmpeg
     """

     # check if valid path
     if not os.path.exists(vid_path):
          print(f"Error: Video file not found at path {vid_path}")
          return None

     # make dir for split video
     os.makedirs(output, exist_ok=True)

     total_duration = get_vid_duration(vid_path=vid_path)
     
     # Video is bad
     if total_duration is None:
          return
     
     # calculate step size
     step = section_duration - overlap
     if step <= 0:
          print("Invalid input, overlap must be less than section_duration")
          return

     # Get the total number of sections in whole video
     num_section = math.ceil((total_duration - overlap) / step)
     if num_section < 1:
          num_section = 1

     print(f"Spltting into {num_section} sections")

     section_meta_data = []

     # loop through every section
     for i in tqdm(range(num_section)):
          start_time = i * step
          # Ensure we don't go past the end of the video
          # The last section might be shorter than section_duration
          if start_time >= total_duration:
               break

          # define the section names and path
          name = os.path.splitext(os.path.basename(vid_path))[0]
          section_filename = f"{name}_section_{i:04d}.mp4"
          section_path = os.path.join(output, section_filename)

          # ffmpeg command for fast seeking and cutting without re-encoding (if possible)
          # Using -ss before -i is faster, but might be slightly less accurate. 
          # For our use case, fast slicing is crucial for 10-hour videos.
          # We re-encode video/audio slightly to ensure exact cuts and avoid broken I-frames at the start.
          command = [
               'ffmpeg',
               '-y', # Overwrite output files without asking
               '-ss', str(start_time),
               '-i', vid_path,
               '-t', str(section_duration),
               '-c:v', 'libx264', # Re-encode to ensure precise cuts
               '-preset', 'fast',
               '-crf', '23',      # Reasonable quality
               '-c:a', 'aac',
               section_path
          ]

          try:
               # Using DEVNULL to keep output clean, but can capture stderr if needed for debugging
               subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
               
               section_meta_data.append({
                    "original_vid_name": os.path.basename(vid_path),
                    "section_id": i,
                    "filename": section_filename,
                    "start_time_global": start_time,
                    "end_time_global": min(start_time + section_duration, total_duration),
                    "duration": min(section_duration, total_duration - start_time)
               })


          except subprocess.CalledProcessError as e:
               print(f"Error processing section {i} : {e}")

          # Save metadata, giving us global time of where each start end is
          metadata_path = os.path.join(output, "metadata.json")
          with open(metadata_path, "w") as f:
               json.dump(section_meta_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split long videos into overlapping section.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", default="section_output", help="Directory to save section")
    parser.add_argument("--duration", type=int, default=300, help="Duration of each section in seconds (default: 300)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between section in seconds (default: 10)")
    
    args = parser.parse_args()
    
    split_video(args.video_path, args.output, args.duration, args.overlap)
