cmake_minimum_required(VERSION 3.10)
project(cpp_dataloader_library C CXX)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
    # -fsanitize=address
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Third-party source files
file(GLOB_RECURSE URING_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/uring/*.c")
file(GLOB_RECURSE URING_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/uring/*.h")
file(GLOB_RECURSE ZSTD_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.c")
file(GLOB_RECURSE ZSTD_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.h")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    enable_language(ASM)
    file(GLOB_RECURSE ZSTD_ASM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/zstd/*.S")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ryml
    ${CMAKE_CURRENT_SOURCE_DIR}/cityhash
    ${CMAKE_CURRENT_SOURCE_DIR}/cpppath
    ${CMAKE_CURRENT_SOURCE_DIR}/worker_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/zstd
    #${Python3_INCLUDE_DIRS}
    #${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    #${Python3_LIBRARIES}
    Threads::Threads
)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "") # remove lib prefix
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Helper macro to add test executables
macro(add_test_executable name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} ${PROJECT_NAME})
endmacro()

# Add test executables
add_test_executable(test_compressor)
add_test_executable(test_mapped_file)
add_test_executable(test_uring_file)
add_test_executable(test_worker_pool)
add_test_executable(test_tools)
add_test_executable(test_data_prep)
add_test_executable(test_data_loader)
