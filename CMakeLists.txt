cmake_minimum_required(VERSION 3.10)
project(distributed_ml)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find packages using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
pkg_check_modules(GRPC REQUIRED grpc++)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Find spdlog
find_package(spdlog REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/network
    ${CMAKE_SOURCE_DIR}/src/utils
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# Add source files
set(SOURCES
    src/main.cpp
    src/core/ml_model.cpp
    src/network/distributed_trainer.cpp
    src/utils/data_loader.cpp
)

# Create executable
add_executable(distributed_ml ${SOURCES})

# Link libraries
target_link_libraries(distributed_ml 
    ${PROTOBUF_LIBRARIES}
    ${GRPC_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    spdlog::spdlog
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Compiler warnings and optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(distributed_ml PRIVATE 
        -Wall 
        -Wextra 
        -pedantic
        -O3
        # Suppress specific warnings for Eigen
        -Wno-class-memaccess
        -Wno-error=class-memaccess
    )
endif()

# Install
install(TARGETS distributed_ml DESTINATION bin)
