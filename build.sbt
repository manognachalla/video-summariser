name := "VideoSummarizer"

version := "1.0"

scalaVersion := "2.12.15"

// Spark dependencies
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided"
)

// Hadoop dependencies
libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-client" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-hdfs" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-common" % "3.3.6" % "provided"
)

// HTTP client for downloads
libraryDependencies += "org.apache.httpcomponents" % "httpclient" % "4.5.14"

// JSON parsing
libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "2.1.1"

// Logging
libraryDependencies += "org.slf4j" % "slf4j-api" % "2.0.9"
libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "2.0.9"

// Testing
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.15" % Test

// Assembly settings for fat JAR
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf" => MergeStrategy.concat
  case x => MergeStrategy.first
}

assembly / assemblyJarName := "video-summarizer-assembly.jar"

// Compiler options
scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xlint"
)