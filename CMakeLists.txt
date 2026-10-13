cmake_minimum_required(VERSION 3.16)
project(AutoVisionADAS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4 /MP)
else()
    add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# Core Library
add_library(autovision_adas_lib STATIC
    src/AdasVisionPipeline.cpp
)

target_link_libraries(autovision_adas_lib PUBLIC
    ${OpenCV_LIBS}
)

# Main Executable Demo
add_executable(autovision_adas_demo
    src/main.cpp
)

target_link_libraries(autovision_adas_demo PRIVATE
    autovision_adas_lib
    ${OpenCV_LIBS}
)

message(STATUS "AutoVision-ADAS configured with OpenCV: ${OpenCV_VERSION}")

