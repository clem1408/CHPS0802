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
include CMakeFiles/runUnitTests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/runUnitTests.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/runUnitTests.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/runUnitTests.dir/flags.make

CMakeFiles/runUnitTests.dir/test/tests.cu.o: CMakeFiles/runUnitTests.dir/flags.make
CMakeFiles/runUnitTests.dir/test/tests.cu.o: CMakeFiles/runUnitTests.dir/includes_CUDA.rsp
CMakeFiles/runUnitTests.dir/test/tests.cu.o: /home/clem/CHPS0802/TP1/test/tests.cu
CMakeFiles/runUnitTests.dir/test/tests.cu.o: CMakeFiles/runUnitTests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/clem/CHPS0802/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/runUnitTests.dir/test/tests.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/runUnitTests.dir/test/tests.cu.o -MF CMakeFiles/runUnitTests.dir/test/tests.cu.o.d -x cu -c /home/clem/CHPS0802/TP1/test/tests.cu -o CMakeFiles/runUnitTests.dir/test/tests.cu.o

CMakeFiles/runUnitTests.dir/test/tests.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/runUnitTests.dir/test/tests.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/runUnitTests.dir/test/tests.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/runUnitTests.dir/test/tests.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o: CMakeFiles/runUnitTests.dir/flags.make
CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o: CMakeFiles/runUnitTests.dir/includes_CUDA.rsp
CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o: /home/clem/CHPS0802/TP1/src/prac1b_impl.cu
CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o: CMakeFiles/runUnitTests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/clem/CHPS0802/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o -MF CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o.d -x cu -c /home/clem/CHPS0802/TP1/src/prac1b_impl.cu -o CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o

CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target runUnitTests
runUnitTests_OBJECTS = \
"CMakeFiles/runUnitTests.dir/test/tests.cu.o" \
"CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o"

# External object files for target runUnitTests
runUnitTests_EXTERNAL_OBJECTS =

runUnitTests: CMakeFiles/runUnitTests.dir/test/tests.cu.o
runUnitTests: CMakeFiles/runUnitTests.dir/src/prac1b_impl.cu.o
runUnitTests: CMakeFiles/runUnitTests.dir/build.make
runUnitTests: CMakeFiles/runUnitTests.dir/linkLibs.rsp
runUnitTests: CMakeFiles/runUnitTests.dir/objects1.rsp
runUnitTests: CMakeFiles/runUnitTests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/clem/CHPS0802/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable runUnitTests"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runUnitTests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/runUnitTests.dir/build: runUnitTests
.PHONY : CMakeFiles/runUnitTests.dir/build

CMakeFiles/runUnitTests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/runUnitTests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/runUnitTests.dir/clean

CMakeFiles/runUnitTests.dir/depend:
	cd /home/clem/CHPS0802/TP1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/clem/CHPS0802/TP1 /home/clem/CHPS0802/TP1 /home/clem/CHPS0802/TP1/build /home/clem/CHPS0802/TP1/build /home/clem/CHPS0802/TP1/build/CMakeFiles/runUnitTests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/runUnitTests.dir/depend

