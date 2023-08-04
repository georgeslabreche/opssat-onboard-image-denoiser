cmake_minimum_required(VERSION 3.16)

# set the compiler environment string argument
set(TARGET "dev" CACHE STRING "Build target")

# set arm compiler environment if we are building for the spacecraft
if(${TARGET} STREQUAL "arm")
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR armv7)
  set(CMAKE_C_COMPILER "/usr/bin/arm-linux-gnueabihf-gcc")
  set(CMAKE_CXX_COMPILER "/usr/bin/arm-linux-gnueabihf-g++")
endif()

# enforce C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the project name
project(denoiser)

# set the TensorFlow Lite source directory
set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/tensorflow" CACHE PATH
  "Directory that contains the TensorFlow project" )

# add the TensorFlow Lite subdirectory
add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# add includes
include_directories(stb)

# add compile options
add_compile_options(-Wall -Os)

# add the executable
add_executable(denoiser src/denoiser.cpp)

# set linker options
set_target_properties(denoiser PROPERTIES LINK_FLAGS "-static")

# link to libraries
target_link_libraries(denoiser m)
target_link_libraries(denoiser tensorflow-lite)