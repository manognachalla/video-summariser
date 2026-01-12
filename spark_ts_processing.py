from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import json
import base64

from feature_extraction_models import SimpleCNN, SimpleNLP, AudioProcessor, SimpleASR


def process_time_series():
    spark = SparkSession.builder \
        .appName("VideoTimeSeriesProcessing") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    print("="*70)
    print("TIME SERIES PROCESSING - VIDEO AS TEMPORAL DATA")
    print("="*70)
    print("\n[Step 1] Loading frames from HDFS...")
    
    frames_path = "hdfs://localhost:9000/user/video_project/processed_frames"
    frames_df = spark.read.json(frames_path)
    
    print(f"Loaded {frames_df.count()} frames")
    frames_df.printSchema()
    print("\n[Step 2] Extracting visual features from frames...")
    cnn = SimpleCNN()
    
    def extract_visual_features_udf(frame_data_base64):
        try:
            import cv2
            frame_bytes = base64.b64decode(frame_data_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            features = cnn.extract_features(frame)
            
            return features.tolist()
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return [0.0] * 512 
    
    extract_features_spark = udf(extract_visual_features_udf, ArrayType(FloatType()))
    
    frames_with_features = frames_df.withColumn(
        "visual_features",
        extract_features_spark(col("frame_data"))
    )
    
    print("✓ Visual features extracted")
    print("\n[Step 3] Applying Moving Average Smoothing...")
    
    window_spec = Window.partitionBy("video_name") \
                        .orderBy("timestamp") \
                        .rowsBetween(-1, 1)  # 3-frame window
    
    def compute_moving_avg(features_list):
        if not features_list or len(features_list) == 0:
            return [0.0] * 512
        
        features_array = np.array(features_list)
        avg_features = np.mean(features_array, axis=0)
        
        return avg_features.tolist()
    
    moving_avg_udf = udf(compute_moving_avg, ArrayType(FloatType()))
    smoothed_df = frames_with_features.withColumn(
        "features_in_window",
        collect_list("visual_features").over(window_spec)
    ).withColumn(
        "smoothed_features",
        moving_avg_udf(col("features_in_window"))
    )
    
    print("✓ Moving average smoothing applied")
    
    print("\n[Step 4] Applying Gaussian Smoothing...")
    
    def gaussian_smooth(features_list, sigma=1.0):
        if not features_list or len(features_list) < 3:
            return features_list[len(features_list)//2] if features_list else [0.0]*512
        

        window_size = len(features_list)
        center = window_size // 2
        
        weights = []
        for i in range(window_size):
            dist = abs(i - center)
            weight = np.exp(-(dist**2) / (2 * sigma**2))
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        features_array = np.array(features_list)
        smoothed = np.sum(features_array * weights[:, np.newaxis], axis=0)
        
        return smoothed.tolist()
    
    gaussian_udf = udf(gaussian_smooth, ArrayType(FloatType()))
    
    smoothed_df = smoothed_df.withColumn(
        "gaussian_smoothed_features",
        gaussian_udf(col("features_in_window"))
    )
    
    print("✓ Gaussian smoothing applied")
    
    print("\n[Step 5] Preparing data for clustering...")
    
    def array_to_vector(arr):
        return Vectors.dense(arr)
    
    to_vector_udf = udf(array_to_vector, VectorUDT())
    
    clustering_df = smoothed_df.withColumn(
        "features_vector",
        to_vector_udf(col("gaussian_smoothed_features"))
    )
    
    scaler = StandardScaler(
        inputCol="features_vector",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    
    scaler_model = scaler.fit(clustering_df)
    scaled_df = scaler_model.transform(clustering_df)
    
    print("✓ Features standardized")
    
    print("\n[Step 6] Performing K-Means Clustering...")
    
    total_frames = scaled_df.count()
    num_clusters = max(5, int(total_frames * 0.1)) 
    
    print(f"Using {num_clusters} clusters for {total_frames} frames")
    
    kmeans = KMeans(
        featuresCol="scaled_features",
        predictionCol="cluster",
        k=num_clusters,
        seed=42,
        maxIter=20
    )
    
    kmeans_model = kmeans.fit(scaled_df)
    clustered_df = kmeans_model.transform(scaled_df)
    
    print("✓ Clustering complete")
    print(f"Cluster centers computed: {len(kmeans_model.clusterCenters())}")
    
    print("\n[Step 7] Selecting keyframes from each cluster...")
    
    def compute_distance_to_center(features, center):
        features_array = np.array(features.toArray())
        center_array = np.array(center)
        distance = np.sqrt(np.sum((features_array - center_array) ** 2))
        return float(distance)
    
    centers_broadcast = spark.sparkContext.broadcast(kmeans_model.clusterCenters())
    
    @udf(FloatType())
    def distance_to_center_udf(features, cluster_id):
        centers = centers_broadcast.value
        return compute_distance_to_center(features, centers[int(cluster_id)])
    
    keyframe_candidates = clustered_df.withColumn(
        "distance_to_center",
        distance_to_center_udf(col("scaled_features"), col("cluster"))
    )
    
    window_cluster = Window.partitionBy("cluster").orderBy("distance_to_center")
    
    keyframes = keyframe_candidates.withColumn(
        "rank_in_cluster",
        row_number().over(window_cluster)
    ).filter(col("rank_in_cluster") == 1) 
    
    keyframes = keyframes.orderBy("video_name", "timestamp")
    
    num_keyframes = keyframes.count()
    print(f"✓ Selected {num_keyframes} keyframes")
    
    output_df = keyframes.select(
        "video_name",
        "frame_number",
        "timestamp",
        "cluster",
        "frame_data",
        "gaussian_smoothed_features"
    )
    
    output_path = "hdfs://localhost:9000/user/video_project/keyframes"
    output_df.write.mode("overwrite").json(output_path)
    
    print(f"✓ Keyframes saved to: {output_path}")
    
    print("\n[Step 9] Generating summary statistics...")
    
    stats_df = keyframes.groupBy("video_name").agg(
        count("frame_number").alias("num_keyframes"),
        min("timestamp").alias("first_keyframe_time"),
        max("timestamp").alias("last_keyframe_time"),
        collect_list("cluster").alias("clusters_represented")
    )
    
    original_counts = frames_df.groupBy("video_name").agg(
        count("frame_number").alias("original_frame_count")
    )
    
    stats_with_ratio = stats_df.join(original_counts, "video_name")
    stats_with_ratio = stats_with_ratio.withColumn(
        "compression_ratio",
        col("original_frame_count") / col("num_keyframes")
    )
    
    print("\nSummary Statistics:")
    stats_with_ratio.show(truncate=False)
    
    stats_path = "hdfs://localhost:9000/user/video_project/summary_stats"
    stats_with_ratio.write.mode("overwrite").json(stats_path)
    
    print(f"✓ Statistics saved to: {stats_path}")
    
    print("\n" + "="*70)
    print("TIME SERIES PROCESSING COMPLETE")
    print("="*70)
    print(f"Total frames processed: {total_frames}")
    print(f"Keyframes selected: {num_keyframes}")
    print(f"Average compression: {total_frames/num_keyframes:.2f}x")
    print("="*70)
    
    spark.stop()


if __name__ == "__main__":
    process_time_series()