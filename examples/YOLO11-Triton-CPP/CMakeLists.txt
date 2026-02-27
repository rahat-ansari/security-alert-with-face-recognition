cmake_minimum_required(VERSION 3.5)

project(YOLO11TritonCPP VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally allow external Triton client directory to be provided
option(TRITON_CLIENT_DIR "Path to Triton client installation" "")

if(TRITON_CLIENT_DIR)
    # help CMake locate packages shipped with the client by pointing to their config dirs
    set(Protobuf_DIR "${TRITON_CLIENT_DIR}/protobuf/lib/cmake/protobuf" CACHE PATH "")
    set(gRPC_DIR "${TRITON_CLIENT_DIR}/grpc" CACHE PATH "")
    set(c-ares_DIR "${TRITON_CLIENT_DIR}/c-ares/lib/cmake/c-ares" CACHE PATH "")
    set(TritonClient_DIR "${TRITON_CLIENT_DIR}/lib/cmake/TritonClient" CACHE PATH "")
    set(TritonCommon_DIR "${TRITON_CLIENT_DIR}/lib/cmake/TritonCommon" CACHE PATH "")
else()
    message(STATUS "TRITON_CLIENT_DIR not specified; will attempt to find Triton packages on CMAKE_PREFIX_PATH")
endif()

# try to locate the Triton client packages using find_package
find_package(TritonCommon QUIET)
find_package(TritonClient QUIET)

if(NOT TritonCommon_FOUND OR NOT TritonClient_FOUND)
    message(STATUS "Triton packages not found yet; make sure TRITON_CLIENT_DIR is set or CMAKE_PREFIX_PATH includes the client installation.")
endif()

if(NOT TritonCommon_FOUND)
    message(FATAL_ERROR "TritonCommon package not found. Set -DTRITON_CLIENT_DIR=/path/to/tritonclient or add its lib/cmake folders to CMAKE_PREFIX_PATH")
endif()

if(NOT TritonClient_FOUND)
    message(FATAL_ERROR "TritonClient package not found. Set -DTRITON_CLIENT_DIR=/path/to/tritonclient or add its lib/cmake folders to CMAKE_PREFIX_PATH")
endif()

# Compiler optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c -mavx2 -O3 -ffast-math -march=native")

# OpenCV setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Triton packages
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)

# Project source files
set(PROJECT_SOURCES
    main.cpp
    inference.cpp
    inference.hpp
)

# Define executable target
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories
target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_DIR}/lib
)

# Link libraries
# prefer imported targets provided by the TritonClient/CMake packages
set(triton_libs grpcclient)
if(TritonClient_FOUND)
    list(APPEND triton_libs
        TritonClient::grpcclient
        TritonClient::httpclient
        TritonClient::model_config
        TritonCommon::tritoncommon
    )
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    ${triton_libs}
)
