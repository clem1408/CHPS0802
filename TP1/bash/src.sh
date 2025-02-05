#!/bin/bash

# Enable automate interrupt in case of errors
set -e

# Variables
BUILD_DIR="../build"
EXECUTABLE_NAME="tp1"

# Verify build/ existency and create it if necessary
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found! Creating it..."
    mkdir -p "$BUILD_DIR"
fi

# Go in build/ folder
cd "$BUILD_DIR"

# Step 1: Run cmake
echo "Running cmake..."
cmake ..
echo "CMake configuration successful!"

# Step 2: Run make
echo "Running make..."
make
echo "Make process completed successfully!"

# Step 3: Run the executable
echo "Running the executable..."
if [ -f "./$EXECUTABLE_NAME" ]; then
    ./"$EXECUTABLE_NAME"
else
    echo "Executable $EXECUTABLE_NAME not found!"
    exit 1
fi

echo "Script finished successfully!"
