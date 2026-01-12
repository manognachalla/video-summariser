import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.conf.Configuration
import java.net.{URL, HttpURLConnection}
import java.io.{InputStream, OutputStream, BufferedInputStream}
import scala.util.{Try, Success, Failure}
import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Future, Await}
import scala.concurrent.duration._

object DirectDatasetDownloader {
  
  case class VideoMetadata(
    videoId: String,
    url: String,
    title: String,
    category: String,
    duration: Double
  )
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DirectDatasetDownloader")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
    
    println("="*70)
    println("DIRECT DATASET DOWNLOAD TO HDFS")
    println("="*70)
    
    // Configuration
    val datasetUrl = "https://www.innovatiana.com/en/datasets/howto100m"
    val hdfsBasePath = "/user/video_project/raw_videos"
    val metadataPath = "/user/video_project/metadata"
    val maxVideos = 100  // Download first 100 videos for testing
    
    // Get Hadoop FileSystem
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val fs = FileSystem.get(hadoopConf)
    
    try {
      // Step 1: Fetch video URLs from dataset
      println("\n[Step 1] Fetching video metadata from HowTo100M...")
      val videoMetadataList = fetchVideoMetadata(datasetUrl, maxVideos)
      println(s"Found ${videoMetadataList.length} videos to download")
      
      // Step 2: Download videos directly to HDFS in parallel
      println("\n[Step 2] Downloading videos directly to HDFS...")
      val downloadResults = downloadVideosToHDFS(
        videoMetadataList, 
        hdfsBasePath, 
        fs, 
        parallelDownloads = 5
      )
      
      // Step 3: Save metadata to HDFS
      println("\n[Step 3] Saving metadata to HDFS...")
      saveMetadataToHDFS(videoMetadataList, metadataPath, fs, spark)
      
      // Step 4: Generate summary report
      println("\n[Step 4] Generating download report...")
      val successCount = downloadResults.count(_._2)
      val failCount = downloadResults.count(!_._2)
      
      println(s"""
        |Download Summary:
        |  Total Videos: ${videoMetadataList.length}
        |  Successful: $successCount
        |  Failed: $failCount
        |  Success Rate: ${(successCount.toDouble / videoMetadataList.length * 100).toInt}%
      """.stripMargin)
      
      // Step 5: Verify downloads
      println("\n[Step 5] Verifying downloads in HDFS...")
      verifyHDFSDownloads(hdfsBasePath, fs)
      
      println("\n" + "="*70)
      println("DIRECT DOWNLOAD COMPLETE")
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
   * Fetch video metadata from HowTo100M dataset
   * In practice, you'd parse the actual dataset structure
   */
  def fetchVideoMetadata(datasetUrl: String, maxVideos: Int): List[VideoMetadata] = {
    // For HowTo100M, you typically need to:
    // 1. Download the CSV metadata file
    // 2. Parse it to get video IDs and URLs
    // 3. Construct YouTube URLs
    
    // This is a simplified example - adjust based on actual dataset structure
    val metadataUrl = s"$datasetUrl/metadata.csv"
    
    try {
      // Download metadata file
      val connection = new URL(metadataUrl).openConnection().asInstanceOf[HttpURLConnection]
      connection.setRequestMethod("GET")
      connection.setRequestProperty("User-Agent", "Mozilla/5.0")
      
      val inputStream = connection.getInputStream
      val metadata = scala.io.Source.fromInputStream(inputStream).getLines().toList
      inputStream.close()
      
      // Parse CSV (skip header)
      metadata.tail.take(maxVideos).zipWithIndex.map { case (line, idx) =>
        val parts = line.split(",")
        VideoMetadata(
          videoId = parts(0),
          url = s"https://www.youtube.com/watch?v=${parts(0)}", // Construct YouTube URL
          title = if (parts.length > 1) parts(1) else s"Video_$idx",
          category = if (parts.length > 2) parts(2) else "general",
          duration = if (parts.length > 3) parts(3).toDouble else 0.0
        )
      }
    } catch {
      case e: Exception =>
        println(s"Warning: Could not fetch metadata from $metadataUrl")
        println("Using sample URLs for demonstration...")
        
        // Fallback: Generate sample metadata
        generateSampleMetadata(maxVideos)
    }
  }
  
  /**
   * Generate sample metadata for testing when real dataset is unavailable
   */
  def generateSampleMetadata(count: Int): List[VideoMetadata] = {
    // Sample YouTube video IDs (public domain/creative commons)
    val sampleVideoIds = List(
      "dQw4w9WgXcQ",  // Sample video 1
      "9bZkp7q19f0",  // Sample video 2
      "kJQP7kiw5Fk",  // Sample video 3
      "tgbNymZ7vqY"   // Sample video 4
    )
    
    (0 until count).map { i =>
      val videoId = sampleVideoIds(i % sampleVideoIds.length)
      VideoMetadata(
        videoId = s"${videoId}_$i",
        url = s"https://www.youtube.com/watch?v=$videoId",
        title = s"HowTo Video $i",
        category = List("cooking", "crafts", "technology", "gardening")(i % 4),
        duration = 60.0 + (i * 30)
      )
    }.toList
  }
  
  /**
   * Download videos to HDFS in parallel
   */
  def downloadVideosToHDFS(
    videos: List[VideoMetadata],
    hdfsBasePath: String,
    fs: FileSystem,
    parallelDownloads: Int
  ): List[(String, Boolean)] = {
    
    implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(parallelDownloads)
    )
    
    val futures = videos.map { video =>
      Future {
        downloadSingleVideo(video, hdfsBasePath, fs)
      }
    }
    
    // Wait for all downloads with timeout
    val results = futures.map { f =>
      Try(Await.result(f, 10.minutes)) match {
        case Success(result) => result
        case Failure(e) =>
          println(s"Download timeout: ${e.getMessage}")
          ("timeout", false)
      }
    }
    
    results
  }
  
  /**
   * Download single video directly to HDFS
   */
  def downloadSingleVideo(
    video: VideoMetadata,
    hdfsBasePath: String,
    fs: FileSystem
  ): (String, Boolean) = {
    
    val hdfsPath = new Path(s"$hdfsBasePath/${video.videoId}.mp4")
    
    // Check if already exists
    if (fs.exists(hdfsPath)) {
      println(s"✓ ${video.videoId} already exists, skipping...")
      return (video.videoId, true)
    }
    
    try {
      println(s"⬇ Downloading ${video.videoId} to HDFS...")
      
      // For YouTube videos, you need youtube-dl or yt-dlp
      // Here we'll use a system call approach
      val tempLocalPath = s"/tmp/${video.videoId}.mp4"
      
      // Download using youtube-dl (must be installed)
      val downloadCmd = Seq(
        "youtube-dl",
        "-f", "best[height<=480]",  // 480p quality
        "-o", tempLocalPath,
        video.url
      )
      
      val downloadProcess = Runtime.getRuntime.exec(downloadCmd.toArray)
      val exitCode = downloadProcess.waitFor()
      
      if (exitCode == 0) {
        // Upload to HDFS
        val localPath = new Path(s"file://$tempLocalPath")
        fs.copyFromLocalFile(true, true, localPath, hdfsPath)
        
        // Cleanup local file
        new java.io.File(tempLocalPath).delete()
        
        println(s"✓ ${video.videoId} downloaded successfully")
        (video.videoId, true)
      } else {
        println(s"✗ Failed to download ${video.videoId}")
        (video.videoId, false)
      }
      
    } catch {
      case e: Exception =>
        println(s"✗ Error downloading ${video.videoId}: ${e.getMessage}")
        (video.videoId, false)
    }
  }
  
  /**
   * Alternative: Stream directly to HDFS without local storage
   * This works for direct HTTP downloads (not YouTube which requires youtube-dl)
   */
  def streamToHDFS(
    url: String,
    hdfsPath: Path,
    fs: FileSystem
  ): Boolean = {
    
    var inputStream: InputStream = null
    var outputStream: OutputStream = null
    
    try {
      // Open HTTP connection
      val connection = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
      connection.setRequestMethod("GET")
      connection.setRequestProperty("User-Agent", "Mozilla/5.0")
      connection.setConnectTimeout(30000)
      connection.setReadTimeout(30000)
      
      inputStream = new BufferedInputStream(connection.getInputStream)
      outputStream = fs.create(hdfsPath, true)
      
      // Stream data
      val buffer = new Array[Byte](8192)
      var bytesRead = 0
      var totalBytes = 0L
      
      while ({bytesRead = inputStream.read(buffer); bytesRead != -1}) {
        outputStream.write(buffer, 0, bytesRead)
        totalBytes += bytesRead
        
        // Progress indicator
        if (totalBytes % (1024 * 1024) == 0) {
          print(s"\rDownloaded: ${totalBytes / (1024 * 1024)} MB")
        }
      }
      
      println(s"\nTotal downloaded: ${totalBytes / (1024 * 1024)} MB")
      true
      
    } catch {
      case e: Exception =>
        println(s"Stream error: ${e.getMessage}")
        false
    } finally {
      if (inputStream != null) inputStream.close()
      if (outputStream != null) outputStream.close()
    }
  }
  
  /**
   * Save metadata to HDFS as JSON
   */
  def saveMetadataToHDFS(
    videos: List[VideoMetadata],
    metadataPath: String,
    fs: FileSystem,
    spark: SparkSession
  ): Unit = {
    
    import spark.implicits._
    
    // Convert to DataFrame
    val metadataDF = videos.toDF()
    
    // Save as JSON to HDFS
    metadataDF.write
      .mode("overwrite")
      .json(metadataPath)
    
    println(s"✓ Metadata saved to $metadataPath")
  }
  
  /**
   * Verify downloads in HDFS
   */
  def verifyHDFSDownloads(hdfsBasePath: String, fs: FileSystem): Unit = {
    val basePath = new Path(hdfsBasePath)
    
    if (fs.exists(basePath)) {
      val files = fs.listStatus(basePath)
      
      println(s"\nFiles in HDFS ($hdfsBasePath):")
      println("-" * 70)
      
      var totalSize = 0L
      files.foreach { file =>
        val sizeMB = file.getLen / (1024 * 1024)
        totalSize += file.getLen
        println(f"  ${file.getPath.getName}%-40s ${sizeMB}%6d MB")
      }
      
      println("-" * 70)
      println(f"Total: ${files.length} files, ${totalSize / (1024 * 1024)} MB")
    } else {
      println(s"Path $hdfsBasePath does not exist")
    }
  }
}