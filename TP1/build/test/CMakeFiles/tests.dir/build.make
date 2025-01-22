# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/clem/CHPS0802/TP1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/clem/CHPS0802/TP1/build

# Include any dependencies generated for this target.
include test/CMakeFiles/tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/tests.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/tests.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/tests.dir/flags.make

test/CMakeFiles/tests.dir/tests.cu.o: test/CMakeFiles/tests.dir/flags.make
test/CMakeFiles/tests.dir/tests.cu.o: test/CMakeFiles/tests.dir/includes_CUDA.rsp
test/CMakeFiles/tests.dir/tests.cu.o: /home/clem/CHPS0802/TP1/test/tests.cu
test/CMakeFiles/tests.dir/tests.cu.o: test/CMakeFiles/tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/clem/CHPS0802/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test/CMakeFiles/tests.dir/tests.cu.o"
	cd /home/clem/CHPS0802/TP1/build/test && /usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/CMakeFiles/tests.dir/tests.cu.o -MF CMakeFiles/tests.dir/tests.cu.o.d -x cu -c /home/clem/CHPS0802/TP1/test/tests.cu -o CMakeFiles/tests.dir/tests.cu.o

test/CMakeFiles/tests.dir/tests.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/tests.dir/tests.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/CMakeFiles/tests.dir/tests.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/tests.dir/tests.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target tests
tests_OBJECTS = \
"CMakeFiles/tests.dir/tests.cu.o"

# External object files for target tests
tests_EXTERNAL_OBJECTS =

test/tests: test/CMakeFiles/tests.dir/tests.cu.o
test/tests: test/CMakeFiles/tests.dir/build.make
test/tests: /usr/lib/x86_64-linux-gnu/libgtest.a
test/tests: test/CMakeFiles/tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/clem/CHPS0802/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tests"
	cd /home/clem/CHPS0802/TP1/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/tests.dir/build: test/tests
.PHONY : test/CMakeFiles/tests.dir/build

test/CMakeFiles/tests.dir/clean:
	cd /home/clem/CHPS0802/TP1/build/test && $(CMAKE_COMMAND) -P CMakeFiles/tests.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/tests.dir/clean

test/CMakeFiles/tests.dir/depend:
	cd /home/clem/CHPS0802/TP1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/clem/CHPS0802/TP1 /home/clem/CHPS0802/TP1/test /home/clem/CHPS0802/TP1/build /home/clem/CHPS0802/TP1/build/test /home/clem/CHPS0802/TP1/build/test/CMakeFiles/tests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/CMakeFiles/tests.dir/depend

