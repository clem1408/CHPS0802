#!/bin/bash

# Variables
BUILD_DIR="../build"
EXECUTABLE_NAME="tp1"

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