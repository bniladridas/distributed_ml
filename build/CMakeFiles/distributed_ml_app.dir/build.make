# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/niladridas/Documents/kuber/distributed_ml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/niladridas/Documents/kuber/distributed_ml/build

# Include any dependencies generated for this target.
include CMakeFiles/distributed_ml_app.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/distributed_ml_app.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/distributed_ml_app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/distributed_ml_app.dir/flags.make

CMakeFiles/distributed_ml_app.dir/codegen:
.PHONY : CMakeFiles/distributed_ml_app.dir/codegen

CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o: CMakeFiles/distributed_ml_app.dir/flags.make
CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o: /Users/niladridas/Documents/kuber/distributed_ml/src/distributed_trainer.cpp
CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o: CMakeFiles/distributed_ml_app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o -MF CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o.d -o CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o -c /Users/niladridas/Documents/kuber/distributed_ml/src/distributed_trainer.cpp

CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niladridas/Documents/kuber/distributed_ml/src/distributed_trainer.cpp > CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.i

CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niladridas/Documents/kuber/distributed_ml/src/distributed_trainer.cpp -o CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.s

CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o: CMakeFiles/distributed_ml_app.dir/flags.make
CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o: /Users/niladridas/Documents/kuber/distributed_ml/src/task_manager.cpp
CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o: CMakeFiles/distributed_ml_app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o -MF CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o.d -o CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o -c /Users/niladridas/Documents/kuber/distributed_ml/src/task_manager.cpp

CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niladridas/Documents/kuber/distributed_ml/src/task_manager.cpp > CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.i

CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niladridas/Documents/kuber/distributed_ml/src/task_manager.cpp -o CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.s

CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o: CMakeFiles/distributed_ml_app.dir/flags.make
CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o: /Users/niladridas/Documents/kuber/distributed_ml/src/performance_tracker.cpp
CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o: CMakeFiles/distributed_ml_app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o -MF CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o.d -o CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o -c /Users/niladridas/Documents/kuber/distributed_ml/src/performance_tracker.cpp

CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niladridas/Documents/kuber/distributed_ml/src/performance_tracker.cpp > CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.i

CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niladridas/Documents/kuber/distributed_ml/src/performance_tracker.cpp -o CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.s

CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o: CMakeFiles/distributed_ml_app.dir/flags.make
CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o: /Users/niladridas/Documents/kuber/distributed_ml/dashboard/dashboard_server.cpp
CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o: CMakeFiles/distributed_ml_app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o -MF CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o.d -o CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o -c /Users/niladridas/Documents/kuber/distributed_ml/dashboard/dashboard_server.cpp

CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niladridas/Documents/kuber/distributed_ml/dashboard/dashboard_server.cpp > CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.i

CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niladridas/Documents/kuber/distributed_ml/dashboard/dashboard_server.cpp -o CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.s

CMakeFiles/distributed_ml_app.dir/src/main.cpp.o: CMakeFiles/distributed_ml_app.dir/flags.make
CMakeFiles/distributed_ml_app.dir/src/main.cpp.o: /Users/niladridas/Documents/kuber/distributed_ml/src/main.cpp
CMakeFiles/distributed_ml_app.dir/src/main.cpp.o: CMakeFiles/distributed_ml_app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/distributed_ml_app.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/distributed_ml_app.dir/src/main.cpp.o -MF CMakeFiles/distributed_ml_app.dir/src/main.cpp.o.d -o CMakeFiles/distributed_ml_app.dir/src/main.cpp.o -c /Users/niladridas/Documents/kuber/distributed_ml/src/main.cpp

CMakeFiles/distributed_ml_app.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/distributed_ml_app.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niladridas/Documents/kuber/distributed_ml/src/main.cpp > CMakeFiles/distributed_ml_app.dir/src/main.cpp.i

CMakeFiles/distributed_ml_app.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/distributed_ml_app.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niladridas/Documents/kuber/distributed_ml/src/main.cpp -o CMakeFiles/distributed_ml_app.dir/src/main.cpp.s

# Object files for target distributed_ml_app
distributed_ml_app_OBJECTS = \
"CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o" \
"CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o" \
"CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o" \
"CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o" \
"CMakeFiles/distributed_ml_app.dir/src/main.cpp.o"

# External object files for target distributed_ml_app
distributed_ml_app_EXTERNAL_OBJECTS =

distributed_ml_app: CMakeFiles/distributed_ml_app.dir/src/distributed_trainer.cpp.o
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/src/task_manager.cpp.o
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/src/performance_tracker.cpp.o
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/dashboard/dashboard_server.cpp.o
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/src/main.cpp.o
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/build.make
distributed_ml_app: /opt/homebrew/opt/open-mpi/lib/libmpi.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_gapi.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_stitching.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_alphamat.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_aruco.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_bgsegm.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_bioinspired.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_ccalib.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_dnn_objdetect.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_dnn_superres.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_dpm.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_face.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_freetype.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_fuzzy.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_hfs.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_img_hash.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_intensity_transform.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_line_descriptor.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_mcc.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_quality.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_rapid.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_reg.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_rgbd.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_saliency.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_sfm.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_signal.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_stereo.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_structured_light.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_superres.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_surface_matching.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_tracking.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_videostab.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_viz.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_wechat_qrcode.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_xfeatures2d.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_xobjdetect.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_xphoto.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/cpprestsdk/lib/libcpprest.2.10.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_log_setup.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_shape.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_highgui.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_datasets.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_plot.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_text.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_ml.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_phase_unwrapping.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_optflow.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_ximgproc.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_video.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_videoio.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_imgcodecs.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_objdetect.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_calib3d.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_dnn.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_features2d.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_flann.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_photo.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_imgproc.4.11.0.dylib
distributed_ml_app: /opt/homebrew/opt/opencv/lib/libopencv_core.4.11.0.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_log.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_thread.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_filesystem.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_atomic.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_chrono.dylib
distributed_ml_app: /opt/homebrew/Cellar/boost/1.87.0/lib/libboost_system.dylib
distributed_ml_app: CMakeFiles/distributed_ml_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable distributed_ml_app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/distributed_ml_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/distributed_ml_app.dir/build: distributed_ml_app
.PHONY : CMakeFiles/distributed_ml_app.dir/build

CMakeFiles/distributed_ml_app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/distributed_ml_app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/distributed_ml_app.dir/clean

CMakeFiles/distributed_ml_app.dir/depend:
	cd /Users/niladridas/Documents/kuber/distributed_ml/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/niladridas/Documents/kuber/distributed_ml /Users/niladridas/Documents/kuber/distributed_ml /Users/niladridas/Documents/kuber/distributed_ml/build /Users/niladridas/Documents/kuber/distributed_ml/build /Users/niladridas/Documents/kuber/distributed_ml/build/CMakeFiles/distributed_ml_app.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/distributed_ml_app.dir/depend

