# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.5/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.5/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build

# Include any dependencies generated for this target.
include CMakeFiles/test_tracking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_tracking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_tracking.dir/flags.make

CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o: CMakeFiles/test_tracking.dir/flags.make
CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o: /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o -c /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp

CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp > CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.i

CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp -o CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.s

# Object files for target test_tracking
test_tracking_OBJECTS = \
"CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o"

# External object files for target test_tracking
test_tracking_EXTERNAL_OBJECTS =

../bin/test_tracking: CMakeFiles/test_tracking.dir/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/src/test_tracking.cpp.o
../bin/test_tracking: CMakeFiles/test_tracking.dir/build.make
../bin/test_tracking: /usr/local/lib/libopencv_dnn.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_gapi.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_highgui.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_ml.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_objdetect.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_photo.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_stitching.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_video.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_videoio.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_imgcodecs.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_calib3d.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_features2d.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_flann.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_imgproc.4.3.0.dylib
../bin/test_tracking: /usr/local/lib/libopencv_core.4.3.0.dylib
../bin/test_tracking: CMakeFiles/test_tracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test_tracking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_tracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_tracking.dir/build: ../bin/test_tracking

.PHONY : CMakeFiles/test_tracking.dir/build

CMakeFiles/test_tracking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_tracking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_tracking.dir/clean

CMakeFiles/test_tracking.dir/depend:
	cd /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build /Users/zhangjingwen/Downloads/liudong/pro/feature_tracker/build_pc/build/CMakeFiles/test_tracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_tracking.dir/depend
