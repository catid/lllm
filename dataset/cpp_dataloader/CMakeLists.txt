cmake_minimum_required(VERSION 3.10)
project(cpp_dataloader C CXX)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

set(CMAKE_CXX_STANDARD 11)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(PythonLibs REQUIRED)

file(GLOB_RECURSE ZSTD_SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.c"
)

file(GLOB_RECURSE ZSTD_HEADER_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.h"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    enable_language(ASM)
    file(GLOB_RECURSE ZSTD_ASM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.S")
    list(APPEND ZSTD_SRC_FILES ${ZSTD_ASM_FILES})
endif()

add_library(${PROJECT_NAME} SHARED
    dataloader.cpp
    mapped_file.cpp
    mapped_file.hpp
    compressor.cpp
    compressor.hpp
    ${ZSTD_SRC_FILES}
    ${ZSTD_HEADER_FILES}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} ${ZSTD_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} Threads::Threads)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

# Test executable test_compressor
add_executable(test_compressor tests/test_compressor.cpp)
target_link_libraries(test_compressor ${PROJECT_NAME})

# Test executable test_mapped_file
add_executable(test_mapped_file tests/test_mapped_file.cpp)
target_link_libraries(test_mapped_file ${PROJECT_NAME})
