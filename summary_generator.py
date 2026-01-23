#!/usr/bin/env python3
"""
Final Video Summary Generation
Stitch keyframes together to create summary video
"""

from pyspark.sql import SparkSession
import subprocess
import tempfile
import os
import cv2
import numpy as np
import base64
from pymongo import MongoClient


def generate_summary_videos():
    """
    Generate final summary videos from keyframes
    """
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("VideoSummaryGeneration") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Initialize MongoDB
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['video_db']
    
    print("="*70)
    print("FINAL VIDEO SUMMARY GENERATION")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Read Keyframes from HDFS
    # ========================================================================
    print("\n[Step 1] Loading keyframes from HDFS...")
    
    keyframes_path = "hdfs://localhost:9000/user/video_project/keyframes"
    keyframes_df = spark.read.json(keyframes_path)
    
    # Group by video
    video_groups = keyframes_df.groupBy("video_name").count()
    videos = [row.video_name for row in video_groups.collect()]
    
    print(f"Found {len(videos)} videos to process")
    
    # ========================================================================
    # STEP 2: Generate Summary for Each Video
    # ========================================================================
    
    for video_idx, video_name in enumerate(videos, 1):
        print(f"\n[Video {video_idx}/{len(videos)}] Processing {video_name}")
        print("-" * 70)
        
        # Get keyframes for this video
        video_keyframes = keyframes_df.filter(
            keyframes_df.video_name == video_name
        ).orderBy("timestamp").collect()
        
        print(f"Keyframes to process: {len(video_keyframes)}")
        
        if len(video_keyframes) == 0:
            print("⚠ No keyframes found, skipping...")
            continue
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            # ================================================================
            # STEP 2.1: Save Keyframes as Images
            # ================================================================
            print("  Saving keyframes as images...")
            
            frame_paths = []
            for i, keyframe in enumerate(video_keyframes):
                # Decode base64 frame data
                frame_data = base64.b64decode(keyframe.frame_data)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Add timestamp overlay
                timestamp_text = f"t={keyframe.timestamp:.1f}s"
                cv2.putText(
                    frame, timestamp_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            print(f"  ✓ Saved {len(frame_paths)} frames")
            
            # ================================================================
            # STEP 2.2: Create Video from Keyframes
            # ================================================================
            print("  Creating summary video...")
            
            # Output video path
            summary_video_name = video_name.replace('.mp4', '_summary.mp4')
            local_output = os.path.join(temp_dir, summary_video_name)
            
            # Create video using FFmpeg
            # Each keyframe shown for 2 seconds
            frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
            
            cmd = [
                'ffmpeg',
                '-framerate', '0.5',  # 1 frame every 2 seconds
                '-i', frame_pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite output
                local_output
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✓ Summary video created")
                
                # ============================================================
                # STEP 2.3: Upload to HDFS
                # ============================================================
                print("  Uploading to HDFS...")
                
                hdfs_output_path = f"/user/video_project/summaries/{summary_video_name}"
                
                upload_cmd = [
                    'hdfs', 'dfs', '-put', '-f',
                    local_output,
                    f"hdfs://localhost:9000{hdfs_output_path}"
                ]
                
                subprocess.run(upload_cmd, check=True)
                print(f"  ✓ Uploaded to HDFS: {hdfs_output_path}")
                
                # ============================================================
                # STEP 2.4: Get Video Stats
                # ============================================================
                video_stats = {
                    'video_name': video_name,
                    'summary_video_name': summary_video_name,
                    'hdfs_path': hdfs_output_path,
                    'num_keyframes': len(video_keyframes),
                    'duration_seconds': len(video_keyframes) * 2,  # 2s per frame
                    'file_size_bytes': os.path.getsize(local_output)
                }
                
                # Get NLP summary from MongoDB
                nlp_result = db.nlp_results.find_one({
                    "filename": {"$regex": video_name.replace('.mp4', '')}
                })
                
                if nlp_result:
                    video_stats['text_summary'] = nlp_result.get('summary', '')
                    video_stats['sentiment'] = nlp_result.get('sentiment', 'neutral')
                    video_stats['keywords'] = nlp_result.get('keywords', [])
                
                # ============================================================
                # STEP 2.5: Store in MongoDB
                # ============================================================
                db.final_summaries.update_one(
                    {'video_name': video_name},
                    {'$set': video_stats},
                    upsert=True
                )
                
                print(f"  ✓ Metadata stored in MongoDB")
                
            else:
                print(f"  ✗ Error creating video: {result.stderr}")
        
        except Exception as e:
            print(f"  ✗ Error processing {video_name}: {e}")
        
        finally:
            # Cleanup temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    # ========================================================================
    # STEP 3: Generate HTML Report
    # ========================================================================
    print("\n[Step 3] Generating HTML report...")
    
    html_report = generate_html_report(db)
    
    # Save locally
    report_path = "/tmp/video_summary_report.html"
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    print(f"✓ HTML report saved to: {report_path}")
    
    # Upload to HDFS
    hdfs_report_path = "/user/video_project/summary_report.html"
    subprocess.run([
        'hdfs', 'dfs', '-put', '-f',
        report_path,
        f"hdfs://localhost:9000{hdfs_report_path}"
    ])
    
    print(f"✓ Report uploaded to HDFS: {hdfs_report_path}")
    
    # ========================================================================
    # STEP 4: Final Summary
    # ========================================================================
    print("\n" + "="*70)
    print("VIDEO SUMMARY GENERATION COMPLETE")
    print("="*70)
    
    final_count = db.final_summaries.count_documents({})
    print(f"Total videos processed: {final_count}")
    
    # Display summary statistics
    total_keyframes = 0
    total_duration = 0
    
    for summary in db.final_summaries.find():
        total_keyframes += summary.get('num_keyframes', 0)
        total_duration += summary.get('duration_seconds', 0)
    
    print(f"Total keyframes: {total_keyframes}")
    print(f"Total summary duration: {total_duration}s ({total_duration/60:.1f}m)")
    print(f"\nHTML Report: {report_path}")
    print("="*70)
    
    # Cleanup
    mongo_client.close()
    spark.stop()


def generate_html_report(db):
    """
    Generate HTML report of all video summaries
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Summary Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .video-summary {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
        }
        .video-title {
            font-size: 20px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .stat-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .sentiment {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .positive { background: #4CAF50; color: white; }
        .negative { background: #f44336; color: white; }
        .neutral { background: #9E9E9E; color: white; }
        .keywords {
            margin: 15px 0;
        }
        .keyword-tag {
            display: inline-block;
            background: #E3F2FD;
            color: #1976D2;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 14px;
        }
        .summary-text {
            background: white;
            padding: 15px;
            border-radius: 5px;
            line-height: 1.6;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Summary Report</h1>
        <p>Generated using HDFS + Spark + MongoDB pipeline with from-scratch ML models</p>
"""
    
    # Get all video summaries
    summaries = db.final_summaries.find()
    
    for summary in summaries:
        video_name = summary.get('video_name', 'Unknown')
        num_keyframes = summary.get('num_keyframes', 0)
        duration = summary.get('duration_seconds', 0)
        sentiment = summary.get('sentiment', 'neutral')
        text_summary = summary.get('text_summary', 'No summary available')
        keywords = summary.get('keywords', [])
        
        html += f"""
        <div class="video-summary">
            <div class="video-title">{video_name}</div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Keyframes</div>
                    <div class="stat-value">{num_keyframes}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Duration</div>
                    <div class="stat-value">{duration}s</div>
                </div>
            </div>
            
            <div class="sentiment {sentiment}">{sentiment.upper()}</div>
            
            <div class="summary-text">
                <strong>Summary:</strong><br>
                {text_summary}
            </div>
            
            <div class="keywords">
                <strong>Keywords:</strong><br>
"""
        
        for kw in keywords[:10]:
            word = kw.get('word', '')
            freq = kw.get('frequency', 0)
            html += f'<span class="keyword-tag">{word} ({freq})</span>'
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    generate_summary_videos()