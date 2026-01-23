#!/bin/bash

################################################################################
# Complete Project Setup Script
# Sets up Scala + Python integrated video summarizer
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_NAME="video_summarizer"
PROJECT_DIR="$HOME/$PROJECT_NAME"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  VIDEO SUMMARIZER - COMPLETE PROJECT SETUP${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "This script will set up the complete project with:"
echo "  • Scala preprocessing components"
echo "  • Python ML pipeline"
echo "  • Integration scripts"
echo "  • All dependencies"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

################################################################################
# Step 1: Create Project Structure
################################################################################

echo -e "\n${YELLOW}[Step 1] Creating project structure...${NC}"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directory structure
mkdir -p scala/src/main/scala
mkdir -p scala/project
mkdir -p python
mkdir -p scripts
mkdir -p config
mkdir -p docs

echo -e "${GREEN}✓ Project structure created at $PROJECT_DIR${NC}"

################################################################################
# Step 2: Setup Scala Project
################################################################################

echo -e "\n${YELLOW}[Step 2] Setting up Scala project...${NC}"

# Create build.sbt
cat > scala/build.sbt << 'EOF'
name := "VideoSummarizer"
version := "1.0"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided",
  "org.apache.hadoop" % "hadoop-client" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-hdfs" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-common" % "3.3.6" % "provided",
  "org.apache.httpcomponents" % "httpclient" % "4.5.14",
  "org.scala-lang.modules" %% "scala-parser-combinators" % "2.1.1"
)

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf" => MergeStrategy.concat
  case x => MergeStrategy.first
}

assembly / assemblyJarName := "video-summarizer-assembly.jar"
EOF

# Create build.properties
cat > scala/project/build.properties << 'EOF'
sbt.version=1.9.7
EOF

# Create plugins.sbt
cat > scala/project/plugins.sbt << 'EOF'
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.5")
EOF

echo -e "${GREEN}✓ Scala project files created${NC}"

################################################################################
# Step 3: Install Dependencies
################################################################################

echo -e "\n${YELLOW}[Step 3] Installing dependencies...${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Install SBT
if ! command -v sbt &> /dev/null; then
    echo "Installing SBT..."
    if [ "$OS" = "linux" ]; then
        echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
        curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
        sudo apt-get update
        sudo apt-get install -y sbt
    else
        brew install sbt
    fi
else
    echo -e "${GREEN}✓ SBT already installed${NC}"
fi

# Install youtube-dl/yt-dlp
if ! command -v yt-dlp &> /dev/null; then
    echo "Installing yt-dlp..."
    sudo pip3 install yt-dlp
else
    echo -e "${GREEN}✓ yt-dlp already installed${NC}"
fi

# Install FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    if [ "$OS" = "linux" ]; then
        sudo apt-get install -y ffmpeg
    else
        brew install ffmpeg
    fi
else
    echo -e "${GREEN}✓ FFmpeg already installed${NC}"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install pyspark pymongo opencv-python numpy pandas scikit-learn

echo -e "${GREEN}✓ All dependencies installed${NC}"

################################################################################
# Step 4: Download Project Files from Artifacts
################################################################################

echo -e "\n${YELLOW}[Step 4] Setting up project files...${NC}"

cat > instructions.txt << 'EOF'
MANUAL STEP REQUIRED:
====================

Please copy the following files from the artifacts to their respective locations:

Scala Files (copy to scala/src/main/scala/):
  1. DirectDatasetDownloader.scala
  2. VideoPreprocessor.scala

Python Files (copy to python/):
  1. upload_videos_to_hdfs.py
  2. spark_frame_extraction.py (or spark_frame_extraction_integrated.py)
  3. feature_extraction_models.py
  4. spark_time_series_processing.py
  5. spark_nlp_processing.py
  6. mongodb_integration.py
  7. generate_final_summary.py

Scripts (copy to scripts/):
  1. integrated_pipeline.sh
  2. run_complete_pipeline.sh

Documentation:
  1. README.md (to project root)
  2. SCALA_INTEGRATION_GUIDE.md (to docs/)
  3. TESTING_AND_VALIDATION.md (to docs/)

After copying the files, run:
  chmod +x scripts/*.sh
  cd scala && sbt assembly

Then you can start using the pipeline!
EOF

cat instructions.txt
echo ""
echo -e "${YELLOW}Please complete the manual steps above.${NC}"

################################################################################
# Step 5: Create Helper Scripts
################################################################################

echo -e "\n${YELLOW}[Step 5] Creating helper scripts...${NC}"

# Create quick start script
cat > scripts/quickstart.sh << 'EOF'
#!/bin/bash

# Quick start with 10 test videos
echo "Starting pipeline with 10 test videos..."
cd "$(dirname "$0")/.."
./scripts/integrated_pipeline.sh 10
EOF

# Create status check script
cat > scripts/check_status.sh << 'EOF'
#!/bin/bash

echo "=== System Status Check ==="
echo ""

echo "HDFS Status:"
hdfs dfsadmin -report 2>/dev/null || echo "  ✗ HDFS not running"

echo ""
echo "MongoDB Status:"
mongo --eval "db.stats()" 2>/dev/null > /dev/null && echo "  ✓ MongoDB running" || echo "  ✗ MongoDB not running"

echo ""
echo "Spark:"
spark-submit --version 2>&1 | head -1

echo ""
echo "Project Files in HDFS:"
hdfs dfs -du -h /user/video_project/ 2>/dev/null | head -10

echo ""
echo "MongoDB Collections:"
mongo video_db --quiet --eval "db.getCollectionNames().forEach(c => print(c + ': ' + db[c].count()))" 2>/dev/null
EOF

# Create cleanup script
cat > scripts/cleanup.sh << 'EOF'
#!/bin/bash

echo "WARNING: This will delete all processed data!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "Cleaning HDFS..."
    hdfs dfs -rm -r /user/video_project/* 2>/dev/null
    
    echo "Cleaning MongoDB..."
    mongo video_db --eval "db.dropDatabase()" 2>/dev/null
    
    echo "Cleanup complete!"
else
    echo "Cleanup cancelled."
fi
EOF

chmod +x scripts/*.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

################################################################################
# Step 6: Create Configuration Files
################################################################################

echo -e "\n${YELLOW}[Step 6] Creating configuration files...${NC}"

cat > config/pipeline.conf << 'EOF'
# Video Summarizer Configuration

[dataset]
url = https://www.innovatiana.com/en/datasets/howto100m
max_videos = 100

[hdfs]
namenode = hdfs://localhost:9000
base_path = /user/video_project

[spark]
master = local[*]
driver_memory = 2g
executor_memory = 4g

[mongodb]
host = localhost
port = 27017
database = video_db

[processing]
fps = 1
frame_size = 224
quality_threshold = 40.0
compression_ratio = 10.0

[quality]
high_score = 80
medium_score = 60
low_score = 40
EOF

echo -e "${GREEN}✓ Configuration files created${NC}"

################################################################################
# Step 7: Initialize Services
################################################################################

echo -e "\n${YELLOW}[Step 7] Initializing services...${NC}"

# Check and start HDFS
if ! hdfs dfsadmin -report &> /dev/null; then
    echo "Starting HDFS..."
    start-dfs.sh
    sleep 5
fi

# Create HDFS directories
echo "Creating HDFS directory structure..."
hdfs dfs -mkdir -p /user/video_project/raw_videos
hdfs dfs -mkdir -p /user/video_project/preprocessed
hdfs dfs -mkdir -p /user/video_project/processed_frames
hdfs dfs -mkdir -p /user/video_project/audio_files
hdfs dfs -mkdir -p /user/video_project/keyframes
hdfs dfs -mkdir -p /user/video_project/summaries
hdfs dfs -mkdir -p /user/video_project/metadata

# Check and start MongoDB
if ! mongo --eval "db.stats()" &> /dev/null; then
    echo "Starting MongoDB..."
    sudo systemctl start mongod
    sleep 3
fi

echo -e "${GREEN}✓ Services initialized${NC}"

################################################################################
# Step 8: Create Documentation
################################################################################

echo -e "\n${YELLOW}[Step 8] Creating project documentation...${NC}"

cat > PROJECT_STRUCTURE.txt << 'EOF'
video_summarizer/
│
├── scala/                          # Scala preprocessing
│   ├── src/main/scala/
│   │   ├── DirectDatasetDownloader.scala
│   │   └── VideoPreprocessor.scala
│   ├── build.sbt
│   ├── project/
│   │   ├── build.properties
│   │   └── plugins.sbt
│   └── target/
│       └── scala-2.12/
│           └── video-summarizer-assembly.jar
│
├── python/                         # Python ML pipeline
│   ├── upload_videos_to_hdfs.py
│   ├── spark_frame_extraction.py
│   ├── spark_frame_extraction_integrated.py
│   ├── feature_extraction_models.py
│   ├── spark_time_series_processing.py
│   ├── spark_nlp_processing.py
│   ├── mongodb_integration.py
│   └── generate_final_summary.py
│
├── scripts/                        # Helper scripts
│   ├── integrated_pipeline.sh
│   ├── run_complete_pipeline.sh
│   ├── quickstart.sh
│   ├── check_status.sh
│   └── cleanup.sh
│
├── config/                         # Configuration
│   └── pipeline.conf
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── SCALA_INTEGRATION_GUIDE.md
│   └── TESTING_AND_VALIDATION.md
│
└── instructions.txt               # Setup instructions
EOF

cat > QUICK_REFERENCE.txt << 'EOF'
QUICK REFERENCE
===============

Build Scala Project:
  cd scala && sbt assembly

Run Complete Pipeline:
  ./scripts/integrated_pipeline.sh 100

Run Quick Test (10 videos):
  ./scripts/quickstart.sh

Check System Status:
  ./scripts/check_status.sh

Clean All Data:
  ./scripts/cleanup.sh

View HDFS Files:
  hdfs dfs -ls -R /user/video_project/

Query MongoDB:
  mongo video_db

View Logs:
  tail -f $SPARK_HOME/logs/*

Monitor Processing:
  watch -n 5 'hdfs dfs -count /user/video_project/raw_videos/'
EOF

echo -e "${GREEN}✓ Documentation created${NC}"

# Final Summary

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  SETUP COMPLETE!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Project Directory: $PROJECT_DIR"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Copy the artifact files as described in instructions.txt"
echo "  2. Build Scala project: cd scala && sbt assembly"
echo "  3. Run quick test: ./scripts/quickstart.sh"
echo "  4. Run full pipeline: ./scripts/integrated_pipeline.sh 100"
echo ""
echo -e "${YELLOW}Documentation:${NC}"
echo "  • PROJECT_STRUCTURE.txt - Directory layout"
echo "  • QUICK_REFERENCE.txt - Common commands"
echo "  • instructions.txt - File placement guide"
echo "  • docs/ - Detailed guides"
echo ""
echo -e "${BLUE}Happy coding! 🚀${NC}"
echo -e "${BLUE}============================================================${NC}"