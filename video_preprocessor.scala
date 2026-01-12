import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.{BufferedReader, InputStreamReader}
import scala.collection.mutable.ListBuffer

/**
 * Video Preprocessing Pipeline in Scala
 * Performs data cleaning, validation, and quality checks
 */
object VideoPreprocessor {
  
  case class VideoInfo(
    videoId: String,
    path: String,
    fileSize: Long,
    isValid: Boolean,
    errorMessage: Option[String]
  )
  
  case class ProcessedVideo(
    videoId: String,
    path: String,
    duration: Double,
    resolution: String,
    fps: Double,
    codec: String,
    bitrate: Long,
    hasAudio: Boolean,
    fileSize: Long,
    qualityScore: Double,
    processingStatus: String
  )
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("VideoPreprocessor")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.shuffle.partitions", "200")
      .getOrCreate()
    
    import spark.implicits._
    
    println("="*70)
    println("VIDEO PREPROCESSING PIPELINE")
    println("="*70)
    
    // Configuration
    val hdfsInputPath = "/user/video_project/raw_videos"
    val hdfsOutputPath = "/user/video_project/preprocessed"
    val metadataPath = "/user/video_project/metadata"
    
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    
    try {
      // Step 1: Scan and validate videos in HDFS
      println("\n[Step 1] Scanning videos in HDFS...")
      val videoInfoList = scanVideosInHDFS(hdfsInputPath, fs)
      println(s"Found ${videoInfoList.length} video files")
      
      val videoInfoDF = videoInfoList.toDF()
      videoInfoDF.show(20, truncate = false)
      
      // Step 2: Extract video metadata using FFprobe
      println("\n[Step 2] Extracting video metadata...")
      val processedVideos = extractVideoMetadata(videoInfoList, fs)
      val processedDF = processedVideos.toDF()
      
      println("\nProcessed Video Statistics:")
      processedDF.describe("duration", "fps", "fileSize").show()
      
      // Step 3: Data Quality Checks
      println("\n[Step 3] Performing quality checks...")
      val qualityCheckedDF = performQualityChecks(processedDF, spark)
      
      // Step 4: Filter and Clean Data
      println("\n[Step 4] Filtering and cleaning data...")
      val cleanedDF = filterAndCleanData(qualityCheckedDF)
      
      // Step 5: Save preprocessed metadata
      println("\n[Step 5] Saving preprocessed metadata...")
      cleanedDF.write
        .mode("overwrite")
        .json(s"$hdfsOutputPath/metadata")
      
      // Step 6: Generate preprocessing report
      println("\n[Step 6] Generating preprocessing report...")
      generatePreprocessingReport(videoInfoDF, cleanedDF, spark)
      
      // Step 7: Create video quality categories
      println("\n[Step 7] Categorizing videos by quality...")
      categorizeByQuality(cleanedDF, hdfsOutputPath)
      
      println("\n" + "="*70)
      println("PREPROCESSING COMPLETE")
      println("="*70)
      
    } catch {
      case e: Exception =>
        println(s"ERROR: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
  
  /**
   * Scan all videos in HDFS and collect basic information
   */
  def scanVideosInHDFS(hdfsPath: String, fs: FileSystem): List[VideoInfo] = {
    val path = new Path(hdfsPath)
    
    if (!fs.exists(path)) {
      println(s"Path $hdfsPath does not exist")
      return List.empty
    }
    
    val files = fs.listStatus(path)
    val videoExtensions = Set(".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm")
    
    files.filter { file =>
      val fileName = file.getPath.getName
      videoExtensions.exists(ext => fileName.toLowerCase.endsWith(ext))
    }.map { file =>
      val videoId = file.getPath.getName.replaceAll("\\.[^.]*$", "")
      VideoInfo(
        videoId = videoId,
        path = file.getPath.toString,
        fileSize = file.getLen,
        isValid = file.getLen > 0, // Basic validation
        errorMessage = if (file.getLen == 0) Some("Empty file") else None
      )
    }.toList
  }
  
  /**
   * Extract detailed metadata from videos using FFprobe
   */
  def extractVideoMetadata(videos: List[VideoInfo], fs: FileSystem): List[ProcessedVideo] = {
    videos.par.map { video =>
      if (!video.isValid) {
        ProcessedVideo(
          videoId = video.videoId,
          path = video.path,
          duration = 0.0,
          resolution = "unknown",
          fps = 0.0,
          codec = "unknown",
          bitrate = 0L,
          hasAudio = false,
          fileSize = video.fileSize,
          qualityScore = 0.0,
          processingStatus = s"invalid: ${video.errorMessage.getOrElse("unknown error")}"
        )
      } else {
        extractMetadataWithFFprobe(video, fs)
      }
    }.toList
  }
  
  /**
   * Use FFprobe to extract video metadata
   */
  def extractMetadataWithFFprobe(video: VideoInfo, fs: FileSystem): ProcessedVideo = {
    try {
      // Download video chunk to analyze (first 5 MB is enough)
      val tempFile = s"/tmp/${video.videoId}_probe.mp4"
      val hdfsPath = new Path(video.path)
      val localPath = new Path(s"file://$tempFile")
      
      // Copy first 5MB to local for analysis
      val inputStream = fs.open(hdfsPath)
      val outputStream = new java.io.FileOutputStream(tempFile)
      val buffer = new Array[Byte](8192)
      var totalRead = 0L
      var bytesRead = 0
      
      while (totalRead < 5 * 1024 * 1024 && {bytesRead = inputStream.read(buffer); bytesRead != -1}) {
        outputStream.write(buffer, 0, bytesRead)
        totalRead += bytesRead
      }
      
      inputStream.close()
      outputStream.close()
      
      // Run FFprobe
      val cmd = Seq(
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        tempFile
      )
      
      val process = Runtime.getRuntime.exec(cmd.toArray)
      val reader = new BufferedReader(new InputStreamReader(process.getInputStream))
      val output = Iterator.continually(reader.readLine()).takeWhile(_ != null).mkString("\n")
      reader.close()
      process.waitFor()
      
      // Parse JSON output
      val metadata = parseFFprobeOutput(output)
      
      // Cleanup
      new java.io.File(tempFile).delete()
      
      // Calculate quality score
      val qualityScore = calculateQualityScore(
        metadata.duration,
        metadata.resolution,
        metadata.fps,
        metadata.bitrate
      )
      
      ProcessedVideo(
        videoId = video.videoId,
        path = video.path,
        duration = metadata.duration,
        resolution = metadata.resolution,
        fps = metadata.fps,
        codec = metadata.codec,
        bitrate = metadata.bitrate,
        hasAudio = metadata.hasAudio,
        fileSize = video.fileSize,
        qualityScore = qualityScore,
        processingStatus = "success"
      )
      
    } catch {
      case e: Exception =>
        println(s"Error processing ${video.videoId}: ${e.getMessage}")
        ProcessedVideo(
          videoId = video.videoId,
          path = video.path,
          duration = 0.0,
          resolution = "unknown",
          fps = 0.0,
          codec = "unknown",
          bitrate = 0L,
          hasAudio = false,
          fileSize = video.fileSize,
          qualityScore = 0.0,
          processingStatus = s"error: ${e.getMessage}"
        )
    }
  }
  
  /**
   * Parse FFprobe JSON output
   */
  def parseFFprobeOutput(jsonOutput: String): ProcessedVideo = {
    import scala.util.parsing.json._
    
    try {
      val json = JSON.parseFull(jsonOutput)
      json match {
        case Some(map: Map[String, Any]) =>
          val format = map.get("format").asInstanceOf[Option[Map[String, Any]]]
          val streams = map.get("streams").asInstanceOf[Option[List[Map[String, Any]]]]
          
          val duration = format.flatMap(_.get("duration").asInstanceOf[Option[String]]).map(_.toDouble).getOrElse(0.0)
          val bitrate = format.flatMap(_.get("bit_rate").asInstanceOf[Option[String]]).map(_.toLong).getOrElse(0L)
          
          // Find video stream
          val videoStream = streams.flatMap(_.find(s => 
            s.get("codec_type").asInstanceOf[Option[String]].contains("video")
          ))
          
          val width = videoStream.flatMap(_.get("width").asInstanceOf[Option[Double]]).map(_.toInt).getOrElse(0)
          val height = videoStream.flatMap(_.get("height").asInstanceOf[Option[Double]]).map(_.toInt).getOrElse(0)
          val resolution = s"${width}x${height}"
          
          val fpsStr = videoStream.flatMap(_.get("r_frame_rate").asInstanceOf[Option[String]]).getOrElse("0/1")
          val fpsParts = fpsStr.split("/")
          val fps = if (fpsParts.length == 2 && fpsParts(1).toInt != 0) 
            fpsParts(0).toDouble / fpsParts(1).toDouble 
          else 0.0
          
          val codec = videoStream.flatMap(_.get("codec_name").asInstanceOf[Option[String]]).getOrElse("unknown")
          
          // Check for audio stream
          val hasAudio = streams.exists(_.exists(s => 
            s.get("codec_type").asInstanceOf[Option[String]].contains("audio")
          ))
          
          ProcessedVideo(
            videoId = "",
            path = "",
            duration = duration,
            resolution = resolution,
            fps = fps,
            codec = codec,
            bitrate = bitrate,
            hasAudio = hasAudio,
            fileSize = 0L,
            qualityScore = 0.0,
            processingStatus = "parsed"
          )
          
        case _ =>
          throw new Exception("Invalid JSON format")
      }
    } catch {
      case e: Exception =>
        println(s"JSON parsing error: ${e.getMessage}")
        ProcessedVideo("", "", 0.0, "unknown", 0.0, "unknown", 0L, false, 0L, 0.0, "parse_error")
    }
  }
  
  /**
   * Calculate quality score based on video properties
   */
  def calculateQualityScore(
    duration: Double,
    resolution: String,
    fps: Double,
    bitrate: Long
  ): Double = {
    
    var score = 0.0
    
    // Duration score (prefer 30s - 600s)
    if (duration >= 30 && duration <= 600) score += 25.0
    else if (duration > 600 && duration <= 1800) score += 20.0
    else if (duration > 0 && duration < 30) score += 10.0
    
    // Resolution score
    val resolutionScore = resolution match {
      case r if r.contains("1920x1080") || r.contains("1080") => 30.0
      case r if r.contains("1280x720") || r.contains("720") => 25.0
      case r if r.contains("640x480") || r.contains("480") => 20.0
      case r if r.contains("854x480") => 20.0
      case _ => 10.0
    }
    score += resolutionScore
    
    // FPS score
    if (fps >= 24 && fps <= 30) score += 20.0
    else if (fps > 30 && fps <= 60) score += 25.0
    else if (fps > 0) score += 10.0
    
    // Bitrate score (in kbps)
    val bitrateKbps = bitrate / 1000
    if (bitrateKbps >= 500 && bitrateKbps <= 5000) score += 25.0
    else if (bitrateKbps > 5000) score += 20.0
    else if (bitrateKbps > 0) score += 10.0
    
    score
  }
  
  /**
   * Perform quality checks on processed videos
   */
  def performQualityChecks(df: DataFrame, spark: SparkSession): DataFrame = {
    import spark.implicits._
    
    df.withColumn("quality_level", 
      when($"qualityScore" >= 80, "high")
        .when($"qualityScore" >= 60, "medium")
        .when($"qualityScore" >= 40, "low")
        .otherwise("poor")
    )
    .withColumn("is_valid_duration", 
      $"duration" >= 10 && $"duration" <= 3600
    )
    .withColumn("is_valid_resolution",
      !$"resolution".contains("unknown") && !$"resolution".contains("0x0")
    )
    .withColumn("is_valid_fps",
      $"fps" >= 15 && $"fps" <= 120
    )
    .withColumn("passes_all_checks",
      $"is_valid_duration" && $"is_valid_resolution" && $"is_valid_fps"
    )
  }
  
  /**
   * Filter and clean data
   */
  def filterAndCleanData(df: DataFrame): DataFrame = {
    println("\nData Filtering:")
    
    val totalCount = df.count()
    println(s"Total videos: $totalCount")
    
    // Filter out invalid videos
    val validDF = df.filter(col("processingStatus") === "success")
    val validCount = validDF.count()
    println(s"Valid videos: $validCount")
    
    // Filter by quality
    val qualityDF = validDF.filter(col("passes_all_checks") === true)
    val qualityCount = qualityDF.count()
    println(s"Quality passed: $qualityCount")
    
    // Remove duplicates based on videoId
    val dedupDF = qualityDF.dropDuplicates("videoId")
    val dedupCount = dedupDF.count()
    println(s"After deduplication: $dedupCount")
    
    // Filter by minimum quality score
    val finalDF = dedupDF.filter(col("qualityScore") >= 40.0)
    val finalCount = finalDF.count()
    println(s"Final dataset: $finalCount")
    
    println(s"\nRetention rate: ${(finalCount.toDouble / totalCount * 100).toInt}%")
    
    finalDF
  }
  
  /**
   * Generate preprocessing report
   */
  def generatePreprocessingReport(rawDF: DataFrame, cleanedDF: DataFrame, spark: SparkSession): Unit = {
    import spark.implicits._
    
    println("\n" + "="*70)
    println("PREPROCESSING REPORT")
    println("="*70)
    
    // Basic statistics
    println(s"\nDataset Size:")
    println(s"  Raw videos: ${rawDF.count()}")
    println(s"  Cleaned videos: ${cleanedDF.count()}")
    println(s"  Removed: ${rawDF.count() - cleanedDF.count()}")
    
    // Quality distribution
    println("\nQuality Distribution:")
    cleanedDF.groupBy("quality_level").count().orderBy(desc("count")).show()
    
    // Resolution distribution
    println("\nResolution Distribution:")
    cleanedDF.groupBy("resolution").count().orderBy(desc("count")).show(10)
    
    // Duration statistics
    println("\nDuration Statistics:")
    cleanedDF.describe("duration").show()
    
    // FPS distribution
    println("\nFPS Distribution:")
    cleanedDF.groupBy("fps").count().orderBy(desc("count")).show(10)
    
    // Codec distribution
    println("\nCodec Distribution:")
    cleanedDF.groupBy("codec").count().orderBy(desc("count")).show()
    
    // Audio presence
    println("\nAudio Presence:")
    cleanedDF.groupBy("hasAudio").count().show()
    
    // Average quality score
    val avgQuality = cleanedDF.agg(avg("qualityScore")).first().getDouble(0)
    println(f"\nAverage Quality Score: $avgQuality%.2f / 100")
  }
  
  /**
   * Categorize videos by quality and save to different directories
   */
  def categorizeByQuality(df: DataFrame, basePath: String): Unit = {
    
    // High quality videos (score >= 80)
    val highQuality = df.filter(col("qualityScore") >= 80)
    highQuality.write.mode("overwrite").json(s"$basePath/high_quality")
    println(s"High quality videos: ${highQuality.count()} saved to $basePath/high_quality")
    
    // Medium quality videos (60 <= score < 80)
    val mediumQuality = df.filter(col("qualityScore") >= 60 && col("qualityScore") < 80)
    mediumQuality.write.mode("overwrite").json(s"$basePath/medium_quality")
    println(s"Medium quality videos: ${mediumQuality.count()} saved to $basePath/medium_quality")
    
    // Low quality videos (40 <= score < 60)
    val lowQuality = df.filter(col("qualityScore") >= 40 && col("qualityScore") < 60)
    lowQuality.write.mode("overwrite").json(s"$basePath/low_quality")
    println(s"Low quality videos: ${lowQuality.count()} saved to $basePath/low_quality")
  }
}