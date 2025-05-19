cmake_minimum_required(VERSION 3.26)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(syclnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_NVIDIA_GPU "Enable Nvidia GPU" OFF)

include(FetchContent)

# onemath
set(ENABLE_MKLCPU_BACKEND ON)
set(ENABLE_MKLGPU_BACKEND OFF) # TODO
set(ENABLE_ROCBLAS_BACKEND OFF) # TODO
set(ENABLE_CUBLAS_BACKEND ${ENABLE_NVIDIA_GPU})
set(TARGET_DOMAINS blas)
set(BUILD_FUNCTIONAL_TESTS OFF)
set(BUILD_EXAMPLES OFF)
FetchContent_Declare(
        onemath
        GIT_REPOSITORY https://github.com/uxlfoundation/onemath.git
        GIT_TAG v0.7
)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.2
)

FetchContent_MakeAvailable(onemath pybind11)

find_package(Python3 REQUIRED COMPONENTS Development)

if(ENABLE_NVIDIA_GPU)
  set(CMAKE_CXX_FLAGS "-fsycl -O3 -fsycl-targets=spir64,nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_50")
else()
  set(CMAKE_CXX_FLAGS "-fsycl -O3")
endif()

pybind11_add_module(syclnn_python
    src/python_bindings.cpp
)

target_include_directories(syclnn_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(syclnn_python PRIVATE
    onemath
    pybind11::pybind11
    Python3::Python
)

set_target_properties(syclnn_python PROPERTIES
    OUTPUT_NAME "_syclnn"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

target_link_libraries(syclnn_python PRIVATE onemath)

if(SKBUILD_PROJECT_NAME)
    install(
        TARGETS syclnn_python
        DESTINATION ${SKBUILD_PROJECT_NAME}
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/
        DESTINATION ${SKBUILD_PROJECT_NAME}
        FILES_MATCHING
            PATTERN "*.so*"
            PATTERN "*.dylib"
            PATTERN "*.dll"
    )
endif()

add_custom_command(TARGET syclnn_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/lib    ${CMAKE_CURRENT_BINARY_DIR}/python/syclnn)
