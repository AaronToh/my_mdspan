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
)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE my_mdspan)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
