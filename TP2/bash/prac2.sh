#!/bin/bash

# Enable automate interrupt in case of errors
set -e

# Variables
BUILD_DIR="../build"
EXECUTABLE_NAME="prac2"

# Verify build/ existency and create it if necessary
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found! Creating it..."
    mkdir -p "$BUILD_DIR"
fi

# Go in build/ folder
cd "$BUILD_DIR"

# Step 1: Run cmake
echo "Running cmake..."
cd "$BUILD_DIR" || { echo "Build directory not found!"; exit 1; }
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Step 2: Run make
echo "Running make..."
make
if [ $? -ne 0 ]; then
    echo "Make process failed!"
    exit 1
fi

# Step 3: Run the executable
echo "Running the executable..."
if [ -f "./$EXECUTABLE_NAME" ]; then
    ./"$EXECUTABLE_NAME"
else
    echo "Executable $EXECUTABLE_NAME not found!"
    exit 1
fi

# Step 4: Clean the build directory
echo "Cleaning build directory..."
make clean
if [ $? -ne 0 ]; then
    echo "Make clean failed!"
    exit 1
fi

echo "Script finished successfully!"

