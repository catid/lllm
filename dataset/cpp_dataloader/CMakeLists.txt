cmake_minimum_required(VERSION 3.10)
project(cpp_dataloader C CXX)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) #FIXME: Release
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
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB_RECURSE URING_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/uring/*.c"
)

file(GLOB_RECURSE URING_HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/uring/*.h"
)

file(GLOB_RECURSE ZSTD_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.c"
)

file(GLOB_RECURSE ZSTD_HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.h"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    enable_language(ASM)
    file(GLOB_RECURSE ZSTD_ASM_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.S")
    list(APPEND ZSTD_SRC_FILES ${ZSTD_ASM_FILES})
endif()

file(GLOB_RECURSE DATALOADER_HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE DATALOADER_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Project library target
add_library(${PROJECT_NAME} SHARED
    ${DATALOADER_HEADER_FILES}
    ${DATALOADER_SRC_FILES}
    cityhash/city.cc
    cityhash/city.h
    ryml/ryml.cpp
    ryml/ryml.hpp
    cpppath/cpppath.h
    ${ZSTD_SRC_FILES}
    ${ZSTD_HEADER_FILES}
    ${URING_SRC_FILES}
    ${URING_HEADER_FILES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/uring/include
)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ryml
    ${CMAKE_CURRENT_SOURCE_DIR}/cityhash
    ${CMAKE_CURRENT_SOURCE_DIR}/cpppath
    ${CMAKE_CURRENT_SOURCE_DIR}/worker_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/zstd
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Python3_LIBRARIES}
    Threads::Threads
)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "") # remove lib prefix
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Test executable test_compressor
add_executable(test_compressor tests/test_compressor.cpp)
target_link_libraries(test_compressor ${PROJECT_NAME})

# Test executable test_mapped_file
add_executable(test_mapped_file tests/test_mapped_file.cpp)
target_link_libraries(test_mapped_file ${PROJECT_NAME})

# Test executable test_uring_file
add_executable(test_uring_file tests/test_uring_file.cpp)
target_link_libraries(test_uring_file ${PROJECT_NAME})

# Test executable test_worker_pool
add_executable(test_worker_pool tests/test_worker_pool.cpp)
target_link_libraries(test_worker_pool ${PROJECT_NAME})

# Test executable test_tools
add_executable(test_tools tests/test_tools.cpp)
target_link_libraries(test_tools ${PROJECT_NAME})

# Test executable test_data_prep
add_executable(test_data_prep tests/test_data_prep.cpp)
target_link_libraries(test_data_prep ${PROJECT_NAME})

# Test executable test_data_loader
add_executable(test_data_loader tests/test_data_loader.cpp)
target_link_libraries(test_data_loader ${PROJECT_NAME})
