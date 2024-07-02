cmake_minimum_required(VERSION 3.10)
project(cpp_distributed_library C CXX)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    # -fsanitize=address
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Find CUDA without using the deprecated FindCUDA module
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(FATAL_ERROR "No CUDA support found")
endif()

# Find PyTorch installation
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(f'{torch.utils.cmake_prefix_path}/Torch')"
    OUTPUT_VARIABLE Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT Torch_DIR)
    message(FATAL_ERROR "PyTorch installation not found. Make sure it's installed in the current Python environment.")
endif()

message(STATUS "Found PyTorch installation: ${Torch_DIR}")
find_package(Torch REQUIRED)

add_subdirectory(cuSZp)

set(QUICHE_LIB_PATH "${CMAKE_SOURCE_DIR}/quiche/target/release/libquiche.a")
if(NOT EXISTS ${QUICHE_LIB_PATH})
    message(FATAL_ERROR "Quiche library not found at ${QUICHE_LIB_PATH}. Please build it first.")
endif()
add_library(quiche STATIC IMPORTED)
set_target_properties(quiche PROPERTIES IMPORTED_LOCATION ${QUICHE_LIB_PATH})

file(GLOB_RECURSE DISTRIBUTED_HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE DISTRIBUTED_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Project library target
add_library(${PROJECT_NAME} SHARED
    ${DISTRIBUTED_HEADER_FILES}
    ${DISTRIBUTED_SRC_FILES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    cuSZp
    quiche
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
add_test_executable(test_fp_compress)
