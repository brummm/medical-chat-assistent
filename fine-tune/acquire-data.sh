#!/bin/bash

# Define the root directory of the project relative to the script
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Define directories
RAW_DATA_DIR="$PROJECT_ROOT/fine-tune/raw-data"
PYTHON_SCRIPT="$PROJECT_ROOT/fine-tune/format-data.py"

# Create the directory if it doesn't exist
mkdir -p "$RAW_DATA_DIR"

# Clone the MedQuAD repository if not already cloned
if [ ! -d "$RAW_DATA_DIR/.git" ]; then
  echo "Cloning MedQuAD repository into $RAW_DATA_DIR..."
  git clone https://github.com/abachaa/MedQuAD "$RAW_DATA_DIR"
else
  echo "MedQuAD repository already exists."
fi

# Run the Python script for data transformation
echo "Running data transformation script..."
python3 "$PYTHON_SCRIPT"
