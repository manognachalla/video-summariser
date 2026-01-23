#!/usr/bin/env python3
"""
MongoDB Integration
Store video summaries, keyframes metadata, and NLP results
"""

from pyspark.sql import SparkSession
from pymongo import MongoClient
from pyspark.sql.functions import *
import json
from datetime import datetime


def store_results_in_mongodb():
    """
    Read processed results from HDFS and store in MongoDB
    """
    # Initialize Spark with MongoDB connector
    spark = SparkSession.builder \
        .appName("MongoDBIntegration") \
        .config("spark.mongodb.output.uri", 
                "mongodb://localhost:27017/video_db.summaries") \
        .config("spark.jars.packages", 
                "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print("="*70)
    print("MONGODB INTEGRATION - STORING RESULTS")
    print("="*70)
    
    # Also initialize direct MongoDB connection for verification
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['video_db']
    
    # ========================================================================
    # STEP 1: Store Keyframes Metadata
    # ========================================================================
    print("\n[Step 1] Storing keyframes metadata in MongoDB...")
    
    # Read keyframes from HDFS
    keyframes_path = "hdfs://localhost:9000/user/video_project/keyframes"
    keyframes_df = spark.read.json(keyframes_path)
    
    # Add metadata
    keyframes_df = keyframes_df.withColumn(
        "processing_timestamp",
        lit(datetime.now().isoformat())
    ).withColumn(
        "data_type",
        lit("keyframe")
    )
    
    # Write to MongoDB
    keyframes_df.write \
        .format("mongo") \
        .mode("overwrite") \
        .option("database", "video_db") \
        .option("collection", "keyframes") \
        .save()
    
    keyframes_count = db.keyframes.count_documents({})
    print(f"✓ Stored {keyframes_count} keyframes in MongoDB")
    
    # ========================================================================
    # STEP 2: Store NLP Results
    # ========================================================================
    print("\n[Step 2] Storing NLP results in MongoDB...")
    
    # Read NLP results from HDFS
    nlp_path = "hdfs://localhost:9000/user/video_project/nlp_results"
    nlp_df = spark.read.json(nlp_path)
    
    # Add metadata
    nlp_df = nlp_df.withColumn(
        "processing_timestamp",
        lit(datetime.now().isoformat())
    ).withColumn(
        "data_type",
        lit("nlp_analysis")
    )
    
    # Write to MongoDB
    nlp_df.write \
        .format("mongo") \
        .mode("overwrite") \
        .option("database", "video_db") \
        .option("collection", "nlp_results") \
        .save()
    
    nlp_count = db.nlp_results.count_documents({})
    print(f"✓ Stored {nlp_count} NLP results in MongoDB")
    
    # ========================================================================
    # STEP 3: Store Summary Statistics
    # ========================================================================
    print("\n[Step 3] Storing summary statistics in MongoDB...")
    
    # Read statistics from HDFS
    stats_path = "hdfs://localhost:9000/user/video_project/summary_stats"
    stats_df = spark.read.json(stats_path)
    
    # Add metadata
    stats_df = stats_df.withColumn(
        "processing_timestamp",
        lit(datetime.now().isoformat())
    ).withColumn(
        "data_type",
        lit("summary_statistics")
    )
    
    # Write to MongoDB
    stats_df.write \
        .format("mongo") \
        .mode("overwrite") \
        .option("database", "video_db") \
        .option("collection", "summary_statistics") \
        .save()
    
    stats_count = db.summary_statistics.count_documents({})
    print(f"✓ Stored {stats_count} summary statistics in MongoDB")
    
    # ========================================================================
    # STEP 4: Create Comprehensive Video Summary Documents
    # ========================================================================
    print("\n[Step 4] Creating comprehensive video summary documents...")
    
    # Join keyframes with NLP results
    video_summaries = keyframes_df.groupBy("video_name").agg(
        count("frame_number").alias("num_keyframes"),
        collect_list(struct(
            "frame_number", "timestamp", "cluster"
        )).alias("keyframe_details")
    )
    
    # Join with NLP data
    nlp_summary = nlp_df.select(
        col("filename").alias("video_name_nlp"),
        "transcript",
        "summary",
        "sentiment",
        "keywords",
        "duration"
    )
    
    # Clean video names for matching
    nlp_summary = nlp_summary.withColumn(
        "video_name_clean",
        regexp_replace(col("video_name_nlp"), r"hdfs://.*?/", "")
    )
    nlp_summary = nlp_summary.withColumn(
        "video_name_clean",
        regexp_replace(col("video_name_clean"), r"\.wav$", ".mp4")
    )
    
    # Join datasets
    comprehensive_summary = video_summaries.join(
        nlp_summary,
        video_summaries.video_name == nlp_summary.video_name_clean,
        "left"
    )
    
    # Add final metadata
    comprehensive_summary = comprehensive_summary.withColumn(
        "processing_timestamp",
        lit(datetime.now().isoformat())
    ).withColumn(
        "summary_version",
        lit("1.0")
    ).withColumn(
        "processing_pipeline",
        lit("HDFS -> Spark -> MongoDB")
    )
    
    # Select final columns
    final_summary = comprehensive_summary.select(
        "video_name",
        "num_keyframes",
        "keyframe_details",
        "transcript",
        "summary",
        "sentiment",
        "keywords",
        "duration",
        "processing_timestamp",
        "summary_version",
        "processing_pipeline"
    )
    
    # Write to MongoDB
    final_summary.write \
        .format("mongo") \
        .mode("overwrite") \
        .option("database", "video_db") \
        .option("collection", "video_summaries") \
        .save()
    
    summary_count = db.video_summaries.count_documents({})
    print(f"✓ Stored {summary_count} comprehensive video summaries")
    
    # ========================================================================
    # STEP 5: Create Indexes for Efficient Querying
    # ========================================================================
    print("\n[Step 5] Creating MongoDB indexes...")
    
    # Create indexes
    db.keyframes.create_index([("video_name", 1)])
    db.keyframes.create_index([("timestamp", 1)])
    db.nlp_results.create_index([("filename", 1)])
    db.nlp_results.create_index([("sentiment", 1)])
    db.video_summaries.create_index([("video_name", 1)])
    db.summary_statistics.create_index([("video_name", 1)])
    
    print("✓ Indexes created")
    
    # ========================================================================
    # STEP 6: Display Sample Data from MongoDB
    # ========================================================================
    print("\n[Step 6] Displaying sample data from MongoDB...")
    print("-" * 70)
    
    print("\nSample Video Summary:")
    sample_summary = db.video_summaries.find_one()
    if sample_summary:
        print(f"Video: {sample_summary.get('video_name', 'N/A')}")
        print(f"Keyframes: {sample_summary.get('num_keyframes', 0)}")
        print(f"Sentiment: {sample_summary.get('sentiment', 'N/A')}")
        print(f"Summary: {sample_summary.get('summary', 'N/A')[:100]}...")
    
    print("\nSample Keywords:")
    sample_nlp = db.nlp_results.find_one()
    if sample_nlp and 'keywords' in sample_nlp:
        keywords = sample_nlp['keywords'][:5]
        for kw in keywords:
            print(f"  - {kw.get('word', 'N/A')}: {kw.get('frequency', 0)}")
    
    # ========================================================================
    # STEP 7: Generate MongoDB Query Examples
    # ========================================================================
    print("\n[Step 7] Useful MongoDB queries:")
    print("-" * 70)
    
    queries = """
# Find all videos with positive sentiment
db.video_summaries.find({"sentiment": "positive"})

# Find keyframes for a specific video
db.keyframes.find({"video_name": "your_video.mp4"})

# Get videos sorted by number of keyframes
db.video_summaries.find().sort({"num_keyframes": -1})

# Find videos containing specific keywords
db.nlp_results.find({"keywords.word": "machine"})

# Get average compression ratio
db.summary_statistics.aggregate([
    {$group: {_id: null, avg_ratio: {$avg: "$compression_ratio"}}}
])

# Find videos processed in the last hour
db.video_summaries.find({
    "processing_timestamp": {
        $gte: new Date(Date.now() - 3600000).toISOString()
    }
})
"""
    
    print(queries)
    
    # ========================================================================
    # STEP 8: Final Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("MONGODB INTEGRATION COMPLETE")
    print("="*70)
    print(f"Database: video_db")
    print(f"Collections created:")
    print(f"  - keyframes: {keyframes_count} documents")
    print(f"  - nlp_results: {nlp_count} documents")
    print(f"  - summary_statistics: {stats_count} documents")
    print(f"  - video_summaries: {summary_count} documents")
    print("="*70)
    
    # Close connections
    mongo_client.close()
    spark.stop()


def query_mongodb_examples():
    """
    Example queries to retrieve data from MongoDB
    """
    print("\n" + "="*70)
    print("MONGODB QUERY EXAMPLES")
    print("="*70)
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['video_db']
    
    # Example 1: Get all video names
    print("\n1. All processed videos:")
    videos = db.video_summaries.distinct("video_name")
    for video in videos:
        print(f"   - {video}")
    
    # Example 2: Get videos by sentiment
    print("\n2. Videos by sentiment:")
    for sentiment in ['positive', 'neutral', 'negative']:
        count = db.video_summaries.count_documents({"sentiment": sentiment})
        print(f"   {sentiment.capitalize()}: {count}")
    
    # Example 3: Get top keywords across all videos
    print("\n3. Top keywords across all videos:")
    pipeline = [
        {"$unwind": "$keywords"},
        {"$group": {
            "_id": "$keywords.word",
            "total_frequency": {"$sum": "$keywords.frequency"}
        }},
        {"$sort": {"total_frequency": -1}},
        {"$limit": 10}
    ]
    
    top_keywords = db.nlp_results.aggregate(pipeline)
    for kw in top_keywords:
        print(f"   {kw['_id']}: {kw['total_frequency']}")
    
    # Example 4: Get video with most keyframes
    print("\n4. Video with most keyframes:")
    video = db.video_summaries.find_one(
        sort=[("num_keyframes", -1)]
    )
    if video:
        print(f"   {video['video_name']}: {video['num_keyframes']} keyframes")
    
    client.close()


if __name__ == "__main__":
    store_results_in_mongodb()
    query_mongodb_examples()