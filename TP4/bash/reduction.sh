#!/bin/bash

# Enable automate interrupt in case of errors
set -e

# Variables
BUILD_DIR="../build"
TEST_DIR="$BUILD_DIR/src"
EXECUTABLE_NAME="reduction"
EXECUTABLE_PATH="$TEST_DIR/$EXECUTABLE_NAME"

# Verify build/ existency and create it if necessary
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found! Creating it..."
    mkdir -p "$BUILD_DIR"
fi

# Verify build/test/ existency and create it if necessary
if [ ! -d "$TEST_DIR" ]; then
    echo "Test directory not found! Creating it..."
    mkdir -p "$TEST_DIR"
fi

# Go in build/ folder
cd "$BUILD_DIR"

# Step 1: Run cmake
echo "Running cmake..."
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
if [ -f "$EXECUTABLE_PATH" ]; then
    "$EXECUTABLE_PATH"
else
    echo "Executable $EXECUTABLE_NAME not found in $TEST_DIR!"
    exit 1
fi

echo "Script finished successfully!"
