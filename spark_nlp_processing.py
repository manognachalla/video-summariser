from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import subprocess
import tempfile
import os

from feature_extraction_models import SimpleASR, SimpleNLP


def process_audio_nlp():
    spark = SparkSession.builder \
        .appName("VideoNLPProcessing") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    print("="*70)
    print("NLP PROCESSING - AUDIO TRANSCRIPTION & TEXT ANALYSIS")
    print("="*70)
    print("\n[Step 1] Reading audio files from HDFS...")
    
    audio_path = "hdfs://localhost:9000/user/video_project/audio_files"
    audio_rdd = sc.binaryFiles(audio_path)
    
    num_audio_files = audio_rdd.count()
    print(f"Found {num_audio_files} audio files")
    print("\n[Step 2] Transcribing audio using from-scratch ASR...")
    asr = SimpleASR()
    
    def transcribe_audio(audio_tuple):
        filename, audio_binary = audio_tuple
        
        try:
            result = asr.transcribe(audio_binary)
            
            return {
                'filename': filename,
                'transcript': result['transcript'],
                'num_segments': result['num_segments'],
                'duration': result['duration'],
                'mfcc_features': result['mfcc_features']
            }
        except Exception as e:
            print(f"Error transcribing {filename}: {e}")
            return {
                'filename': filename,
                'transcript': f"[Transcription error: {str(e)}]",
                'num_segments': 0,
                'duration': 0.0,
                'mfcc_features': []
            }
    transcriptions_rdd = audio_rdd.map(transcribe_audio)
    transcriptions = transcriptions_rdd.collect()
    
    print(f"✓ Transcribed {len(transcriptions)} audio files")
    transcriptions_df = spark.createDataFrame(transcriptions)
    
    print("\n[Step 3] Extracting keywords and processing text...")
    
    nlp = SimpleNLP()
    
    def extract_keywords_udf(transcript):
        if not transcript or transcript.startswith('['):
            return []
        
        try:
            keywords = nlp.extract_keywords(transcript, top_n=10)
            return [{'word': word, 'frequency': freq} for word, freq in keywords]
        except:
            return []
    
    keywords_udf = udf(extract_keywords_udf, ArrayType(
        StructType([
            StructField('word', StringType()),
            StructField('frequency', IntegerType())
        ])
    ))
    
    nlp_df = transcriptions_df.withColumn(
        "keywords",
        keywords_udf(col("transcript"))
    )
    
    print("✓ Keywords extracted")
    print("\n[Step 4] Computing TF-IDF scores...")
    all_transcripts = nlp_df.select("transcript").rdd.map(lambda r: r[0]).collect()
    valid_transcripts = [t for t in all_transcripts if not t.startswith('[')]
    
    if len(valid_transcripts) > 0:
        tfidf_matrix, vocab = nlp.compute_tfidf(valid_transcripts)
        
        print(f"✓ TF-IDF computed for {len(valid_transcripts)} documents")
        print(f"   Vocabulary size: {len(vocab)}")
        
        def get_top_tfidf_terms(doc_idx, n=5):
            scores = tfidf_matrix[doc_idx]
            top_indices = scores.argsort()[-n:][::-1]
            return [(vocab[i], float(scores[i])) for i in top_indices]
        
        tfidf_results = []
        for i, transcript in enumerate(valid_transcripts):
            top_terms = get_top_tfidf_terms(i)
            tfidf_results.append({
                'transcript': transcript,
                'top_tfidf_terms': [{'term': term, 'score': score} 
                                   for term, score in top_terms]
            })
        
        tfidf_df = spark.createDataFrame(tfidf_results)
        
        nlp_df = nlp_df.join(
            tfidf_df,
            nlp_df.transcript == tfidf_df.transcript,
            'left'
        ).drop(tfidf_df.transcript)
    else:
        print("⚠ No valid transcripts for TF-IDF computation")
    
    print("\n[Step 5] Generating text summaries...")
    
    def generate_summary(transcript, keywords, duration):
        if not transcript or transcript.startswith('['):
            return "No audio content detected."
        
        if keywords and len(keywords) > 0:
            top_keywords = [kw['word'] for kw in keywords[:5]]
            keyword_str = ', '.join(top_keywords)
            
            summary = (f"This {duration:.1f} second audio segment "
                      f"discusses topics related to: {keyword_str}. ")
        else:
            summary = f"Audio segment of {duration:.1f} seconds. "
        
        words = transcript.split()
        num_words = len(words)
        
        summary += f"Approximate word count: {num_words}."
        
        return summary
    
    summary_udf = udf(generate_summary, StringType())
    
    nlp_df = nlp_df.withColumn(
        "summary",
        summary_udf(col("transcript"), col("keywords"), col("duration"))
    )
    
    print("✓ Summaries generated")
    
    print("\n[Step 6] Performing sentiment analysis...")
    
    def simple_sentiment(transcript):
        if not transcript or transcript.startswith('['):
            return 'neutral'
        
        transcript_lower = transcript.lower()
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful',
                         'fantastic', 'love', 'best', 'perfect', 'awesome'}
        negative_words = {'bad', 'poor', 'terrible', 'horrible', 'worst',
                         'awful', 'hate', 'disappointing', 'problem', 'issue'}
        pos_count = sum(1 for word in positive_words if word in transcript_lower)
        neg_count = sum(1 for word in negative_words if word in transcript_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    sentiment_udf = udf(simple_sentiment, StringType())
    
    nlp_df = nlp_df.withColumn(
        "sentiment",
        sentiment_udf(col("transcript"))
    )
    
    print("✓ Sentiment analysis complete")
    print("\n[Step 7] Saving NLP results to HDFS...")
    output_path = "hdfs://localhost:9000/user/video_project/nlp_results"
    nlp_df.write.mode("overwrite").json(output_path)
    print(f"✓ NLP results saved to: {output_path}")
    print("\n[Step 8] NLP Summary Statistics:")
    print("-" * 70)
    print("\nSentiment Distribution:")
    nlp_df.groupBy("sentiment").count().show()
    
    avg_duration = nlp_df.agg(avg("duration")).collect()[0][0]
    print(f"\nAverage audio duration: {avg_duration:.2f} seconds")
    
    total_segments = nlp_df.agg(sum("num_segments")).collect()[0][0]
    print(f"Total speech segments detected: {total_segments}")
    

    print("\nSample NLP Results:")
    nlp_df.select("filename", "summary", "sentiment") \
          .show(5, truncate=True)
    
    print("\n" + "="*70)
    print("NLP PROCESSING COMPLETE")
    print("="*70)
    
    spark.stop()


if __name__ == "__main__":
    process_audio_nlp()