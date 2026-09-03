cmake_minimum_required(VERSION 3.26)

project(syclnn VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------ options
# SYCL implementation: dpcpp (Intel icpx or the open-source intel/llvm clang++) or adaptivecpp.
set(SYCLNN_SYCL_IMPL "auto" CACHE STRING "SYCL implementation: auto | dpcpp | adaptivecpp")
# DPC++ offload targets. Aliases: spir64 (CPU/Intel GPU JIT), nvidia_gpu_sm_XX, amd_gpu_gfxXXXX.
set(SYCLNN_TARGETS "spir64" CACHE STRING "Semicolon-separated -fsycl-targets list (DPC++)")
# AdaptiveCpp targets (generic = SSCP JIT for every backend).
set(SYCLNN_ACPP_TARGETS "generic" CACHE STRING "ACPP_TARGETS for AdaptiveCpp builds")
set(SYCLNN_ONEMATH_ROOT "" CACHE PATH "Prefix of a pre-built oneMath (find_package); empty = FetchContent build")
set(SYCLNN_BLAS_BACKENDS "mklcpu;netlib" CACHE STRING "oneMath backends to enable when building oneMath from source")
set(SYCLNN_ONEMATH_TAG "v0.9" CACHE STRING "oneMath git tag for the FetchContent build")
option(SYCLNN_BUILD_BENCH "Build bench/train_bench (C++ driver)" ON)
option(SYCLNN_BUILD_PYTHON "Build the pybind11 module" ON)
option(SYCLNN_FAST_MATH "Compile with -ffast-math (breaks parity with the reference)" OFF)
set(SYCLNN_MODULE_NAME "_syclnn" CACHE STRING "Python extension module name")

# ------------------------------------------------------------------ SYCL compiler
include(CheckCXXCompilerFlag)
if(SYCLNN_SYCL_IMPL STREQUAL "auto")
  check_cxx_compiler_flag("-fsycl" SYCLNN_HAS_FSYCL)
  if(SYCLNN_HAS_FSYCL)
    set(SYCLNN_SYCL_IMPL dpcpp)
  else()
    find_package(AdaptiveCpp QUIET)
    if(AdaptiveCpp_FOUND)
      set(SYCLNN_SYCL_IMPL adaptivecpp)
    else()
      message(FATAL_ERROR
        "No SYCL compiler: set CMAKE_CXX_COMPILER to icpx / intel-llvm clang++ (DPC++) or to acpp, "
        "or pass -DSYCLNN_SYCL_IMPL=adaptivecpp -DAdaptiveCpp_DIR=<prefix>/lib/cmake/AdaptiveCpp")
    endif()
  endif()
endif()
message(STATUS "syclnn: SYCL implementation = ${SYCLNN_SYCL_IMPL}")

add_library(syclnn_sycl INTERFACE)
if(SYCLNN_SYCL_IMPL STREQUAL "dpcpp")
  string(REPLACE ";" "," _targets "${SYCLNN_TARGETS}")
  target_compile_options(syclnn_sycl INTERFACE -fsycl "-fsycl-targets=${_targets}" -fsycl-unnamed-lambda)
  target_link_options(syclnn_sycl INTERFACE -fsycl "-fsycl-targets=${_targets}")
  set(SYCLNN_TARGET_STRING "${_targets}")
elseif(SYCLNN_SYCL_IMPL STREQUAL "adaptivecpp")
  find_package(AdaptiveCpp REQUIRED)
  set(ONEMATH_SYCL_IMPLEMENTATION adaptivecpp)
  # oneMath's exported targets link ONEMATH::SYCL::SYCL, which its FindCompiler.cmake
  # only defines for DPC++; provide it for AdaptiveCpp consumers.
  if(NOT TARGET ONEMATH::SYCL::SYCL)
    add_library(ONEMATH::SYCL::SYCL INTERFACE IMPORTED)
    set_target_properties(ONEMATH::SYCL::SYCL PROPERTIES INTERFACE_LINK_LIBRARIES AdaptiveCpp::acpp-rt)
  endif()
  set(SYCLNN_TARGET_STRING "acpp:${SYCLNN_ACPP_TARGETS}")
else()
  message(FATAL_ERROR "SYCLNN_SYCL_IMPL must be auto, dpcpp or adaptivecpp")
endif()
if(SYCLNN_FAST_MATH)
  target_compile_options(syclnn_sycl INTERFACE -ffast-math)
endif()

# ------------------------------------------------------------------ oneMath
if(SYCLNN_ONEMATH_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${SYCLNN_ONEMATH_ROOT}")
endif()
find_package(oneMath CONFIG QUIET)
if(oneMath_FOUND)
  message(STATUS "syclnn: using installed oneMath ${oneMath_VERSION} from ${oneMath_DIR}")
  set(SYCLNN_ONEMATH_VERSION "${oneMath_VERSION} ${oneMath_DIR}")
  set(_onemath_libs ONEMATH::onemath)
  foreach(_b IN ITEMS mklcpu netlib generic cublas rocblas)
    if(TARGET ONEMATH::onemath_blas_${_b})
      list(APPEND _onemath_libs ONEMATH::onemath_blas_${_b})
    endif()
  endforeach()
else()
  message(STATUS "syclnn: building oneMath ${SYCLNN_ONEMATH_TAG} from source (backends: ${SYCLNN_BLAS_BACKENDS})")
  include(FetchContent)
  set(TARGET_DOMAINS blas CACHE STRING "" FORCE)
  set(BUILD_FUNCTIONAL_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  foreach(_b IN ITEMS MKLCPU MKLGPU NETLIB GENERIC_BLAS CUBLAS ROCBLAS)
    set(ENABLE_${_b}_BACKEND OFF CACHE BOOL "" FORCE)
  endforeach()
  foreach(_b IN LISTS SYCLNN_BLAS_BACKENDS)
    string(TOUPPER "${_b}" _B)
    if(_B STREQUAL "GENERIC")
      set(_B GENERIC_BLAS)
    endif()
    set(ENABLE_${_B}_BACKEND ON CACHE BOOL "" FORCE)
  endforeach()
  if(SYCLNN_SYCL_IMPL STREQUAL "adaptivecpp")
    set(ONEMATH_SYCL_IMPLEMENTATION adaptivecpp CACHE STRING "" FORCE)
    set(ACPP_TARGETS "${SYCLNN_ACPP_TARGETS}" CACHE STRING "" FORCE)
  endif()
  FetchContent_Declare(onemath
    GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
    GIT_TAG ${SYCLNN_ONEMATH_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(onemath)
  set(SYCLNN_ONEMATH_VERSION "${SYCLNN_ONEMATH_TAG} FetchContent")
  set(_onemath_libs onemath)
  foreach(_b IN LISTS SYCLNN_BLAS_BACKENDS)
    if(TARGET onemath_blas_${_b})
      list(APPEND _onemath_libs onemath_blas_${_b})
    endif()
  endforeach()
endif()

# ------------------------------------------------------------------ the header-only library
add_library(syclnn INTERFACE)
add_library(syclnn::syclnn ALIAS syclnn)
target_include_directories(syclnn INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(syclnn INTERFACE syclnn_sycl ${_onemath_libs})

string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
string(STRIP "${_flags}" _flags)
set(SYCLNN_COMPILER_ID "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_CXX_COMPILER}")
if(SYCLNN_SYCL_IMPL STREQUAL "adaptivecpp")
  set(SYCLNN_COMPILER_ID "AdaptiveCpp ${AdaptiveCpp_VERSION} ${ACPP_COMPILER} / host ${SYCLNN_COMPILER_ID}")
endif()
set(SYCLNN_COMPILE_FLAGS "${_flags}")
configure_file(include/syclnn/build_info.hpp.in "${CMAKE_BINARY_DIR}/generated/syclnn/build_info.hpp" @ONLY)
target_include_directories(syclnn INTERFACE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

function(syclnn_sycl_sources target)
  if(SYCLNN_SYCL_IMPL STREQUAL "adaptivecpp")
    add_sycl_to_target(TARGET ${target} SOURCES ${ARGN})
  endif()
endfunction()

# ------------------------------------------------------------------ Python module
if(SYCLNN_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pb11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_pb11_dir)
      list(APPEND CMAKE_PREFIX_PATH "${_pb11_dir}")
      find_package(pybind11 CONFIG QUIET)
    endif()
  endif()
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v3.1.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(syclnn_python MODULE src/python_bindings.cpp)
  syclnn_sycl_sources(syclnn_python src/python_bindings.cpp)
  target_link_libraries(syclnn_python PRIVATE syclnn)
  target_compile_definitions(syclnn_python PRIVATE SYCLNN_MODULE_NAME=${SYCLNN_MODULE_NAME})
  set_target_properties(syclnn_python PROPERTIES
    OUTPUT_NAME "${SYCLNN_MODULE_NAME}"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
  if(SKBUILD_PROJECT_NAME)
    install(TARGETS syclnn_python DESTINATION ${SKBUILD_PROJECT_NAME})
    if(NOT oneMath_FOUND)
      # bundle the oneMath libraries built from source next to the module
      install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${SKBUILD_PROJECT_NAME}
              FILES_MATCHING PATTERN "*.so*")
    else()
      # point at the installed oneMath instead of copying it
      get_target_property(_onemath_loc ONEMATH::onemath LOCATION)
      get_filename_component(_onemath_dir "${_onemath_loc}" DIRECTORY)
      set_target_properties(syclnn_python PROPERTIES INSTALL_RPATH "$ORIGIN;${_onemath_dir}")
    endif()
  endif()
endif()

# ------------------------------------------------------------------ C++ driver + smoke test
if(SYCLNN_BUILD_BENCH)
  add_executable(train_bench bench/train_bench.cpp)
  syclnn_sycl_sources(train_bench bench/train_bench.cpp)
  target_link_libraries(train_bench PRIVATE syclnn)
  include(CTest)
  add_test(NAME smoke_train COMMAND train_bench --layers 8,16,4 --samples 64 --batch 16 --epochs 2 --check)
endif()

# ------------------------------------------------------------------ install (headers)
install(DIRECTORY include/ DESTINATION include)
