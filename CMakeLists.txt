cmake_minimum_required(VERSION 3.20)
project(my_mdspan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library target
add_library(my_mdspan INTERFACE)
target_include_directories(my_mdspan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tests
enable_testing()

foreach(test_name
    test_extents
    test_layout
    test_mdspan
    test_layout_stride
    test_submdspan
    test_ops
    test_mdarray
    test_dispatch
)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE my_mdspan)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# CUDA targets (only built if CUDA is available)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_executable(test_matmul_cuda cuda/test_matmul.cu cuda/matmul.cu)
    target_include_directories(test_matmul_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
    set_target_properties(test_matmul_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME test_matmul_cuda COMMAND test_matmul_cuda)
endif()
