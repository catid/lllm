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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
    # -fsanitize=address
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
add_test_executable(test_fp_compress)
