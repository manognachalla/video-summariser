"""
Integrated Frame Extraction - Works with Scala Preprocessed Data
Reads from Scala's quality-categorized directories
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import subprocess
import os
import tempfile
import json
import base64
import cv2
import numpy as np

def extract_frames_from_video(video_data_tuple):
    filename, video_binary = video_data_tuple
    frames_extracted = []
    
    quality_category = "unknown"
    if "/high_quality/" in filename:
        quality_category = "high"
    elif "/medium_quality/" in filename:
        quality_category = "medium"
    elif "/low_quality/" in filename:
        quality_category = "low"
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        tmp_video.write(video_binary)
        tmp_video_path = tmp_video.name
    
    temp_frame_dir = tempfile.mkdtemp()
    
    try:
        cmd = [
            'ffmpeg',
            '-i', tmp_video_path,
            '-vf', 'fps=1',
            '-q:v', '2',
            f'{temp_frame_dir}/frame_%04d.jpg'
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        frame_files = sorted([f for f in os.listdir(temp_frame_dir) if f.endswith('.jpg')])
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(temp_frame_dir, frame_file)
            
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                frame_resized = cv2.resize(frame, (224, 224))
                
                frame_bytes = cv2.imencode('.jpg', frame_resized)[1].tobytes()
                
                frames_extracted.append({
                    'video_name': os.path.basename(filename),
                    'video_path': filename,
                    'frame_number': i,
                    'timestamp': i * 1.0,
                    'frame_data': base64.b64encode(frame_bytes).decode('utf-8'),
                    'shape': frame_resized.shape,
                    'quality_category': quality_category
                })
        
        for frame_file in frame_files:
            os.remove(os.path.join(temp_frame_dir, frame_file))
        os.rmdir(temp_frame_dir)
        os.remove(tmp_video_path)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(temp_frame_dir):
            import shutil
            shutil.rmtree(temp_frame_dir)
    
    return frames_extracted


def extract_audio_from_video(video_data_tuple):
    filename, video_binary = video_data_tuple
    
    quality_category = "unknown"
    if "/high_quality/" in filename:
        quality_category = "high"
    elif "/medium_quality/" in filename:
        quality_category = "medium"
    elif "/low_quality/" in filename:
        quality_category = "low"
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        tmp_video.write(video_binary)
        tmp_video_path = tmp_video.name

    audio_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_path = audio_output.name
    audio_output.close()
    
    try:
        cmd = [
            'ffmpeg',
            '-i', tmp_video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        os.remove(tmp_video_path)
        os.remove(audio_path)
        
        return (filename, audio_data, quality_category)
        
    except Exception as e:
        print(f"Error extracting audio from {filename}: {str(e)}")
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return (filename, None, quality_category)


def load_preprocessing_metadata(spark):
    try:
        metadata_path = "hdfs://localhost:9000/user/video_project/preprocessed/metadata"
        metadata_df = spark.read.json(metadata_path)
        print("\nPreprocessing Metadata Loaded:")
        metadata_df.printSchema()
        metadata_df.show(5, truncate=False)
        return metadata_df
    except Exception as e:
        print(f"Could not load preprocessing metadata: {e}")
        return None


def main():
    conf = SparkConf().setAppName("IntegratedFrameExtraction") # Initialize Spark
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "2g")
    
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    print("="*70)
    print("INTEGRATED VIDEO FRAME EXTRACTION")
    print("Reading from Scala-preprocessed directories")
    print("="*70)
    
    preprocessing_metadata = load_preprocessing_metadata(spark)
    
    quality_paths = [
        ("high_quality", "hdfs://localhost:9000/user/video_project/preprocessed/high_quality"),
        ("medium_quality", "hdfs://localhost:9000/user/video_project/preprocessed/medium_quality"),
        ("low_quality", "hdfs://localhost:9000/user/video_project/preprocessed/low_quality")
    ]
    
    all_frames = []
    all_audio = []
    
    for quality_name, quality_path in quality_paths:
        print(f"\n{'='*70}")
        print(f"Processing {quality_name.upper()} videos")
        print(f"{'='*70}")
        
        try:
            videos_rdd = sc.binaryFiles(quality_path)
            num_videos = videos_rdd.count()
            
            if num_videos == 0:
                print(f"No videos found in {quality_name}, skipping...")
                continue
            
            print(f"Found {num_videos} {quality_name} videos")
            
            print(f"\n{'-'*70}")
            print(f"PHASE 1: Extracting Frames from {quality_name} videos")
            print(f"{'-'*70}")
            
            frames_rdd = videos_rdd.flatMap(extract_frames_from_video)
            frames = frames_rdd.collect()
            all_frames.extend(frames)
            
            print(f"✓ Extracted {len(frames)} frames from {quality_name} videos")
            
            print(f"\n{'-'*70}")
            print(f"PHASE 2: Extracting Audio from {quality_name} videos")
            print(f"{'-'*70}")
            
            audio_rdd = videos_rdd.map(extract_audio_from_video)
            audio_data = audio_rdd.collect()
            all_audio.extend(audio_data)
            
            print(f"✓ Extracted audio from {len(audio_data)} {quality_name} videos")
            
        except Exception as e:
            print(f"Error processing {quality_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("SAVING COMBINED RESULTS")
    print(f"{'='*70}")
    
    if len(all_frames) > 0:
        frames_df = spark.createDataFrame(all_frames)
        output_path = "hdfs://localhost:9000/user/video_project/processed_frames"
        frames_df.write.mode("overwrite").json(output_path)
        print(f"✓ Saved {len(all_frames)} frames total to: {output_path}")
        
        print("\nFrames by Quality Category:")
        frames_df.groupBy("quality_category").count().show()
    else:
        print("⚠ No frames extracted")
    
    if len(all_audio) > 0:
        for filename, audio_binary, quality in all_audio:
            if audio_binary:
                video_name = os.path.basename(filename)
                audio_filename = video_name.replace('.mp4', '.wav')
                audio_hdfs_path = f"/user/video_project/audio_files/{audio_filename}"
                
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
                    tmp.write(audio_binary)
                    tmp_path = tmp.name
                
                cmd = ['hdfs', 'dfs', '-put', '-f', tmp_path, 
                       f"hdfs://localhost:9000{audio_hdfs_path}"]
                subprocess.run(cmd)
                os.remove(tmp_path)
                print(f"✓ Saved audio: {audio_filename} (quality: {quality})")
    else:
        print("⚠ No audio extracted")
    
    print(f"\n{'='*70}")
    print("EXTRACTION STATISTICS")
    print(f"{'='*70}")
    
    if preprocessing_metadata:
        print("\nComparison with Preprocessing Phase:")
        print(f"Videos preprocessed: {preprocessing_metadata.count()}")
        print(f"Videos processed here: {len(all_audio)}")
        print(f"Total frames extracted: {len(all_frames)}")
        
        if len(all_audio) > 0:
            avg_frames = len(all_frames) / len(all_audio)
            print(f"Average frames per video: {avg_frames:.2f}")
    
    print("\n" + "="*70)
    print("INTEGRATED FRAME EXTRACTION COMPLETE")
    print("="*70)
    sc.stop()


if __name__ == "__main__":
    main()