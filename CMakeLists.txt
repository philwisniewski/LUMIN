cmake_minimum_required(VERSION 3.16)
project(LUMIN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options for backends
option(ENABLE_MPI "Enable MPI backend" ON)
option(ENABLE_CUDA "Enable CUDA backend" ON)
option(ENABLE_OPENMP "Enable OpenMP backend" ON)
option(ENABLE_TESTS "Enable building tests" OFF)

# include
include_directories(${PROJECT_SOURCE_DIR}/include)

# MPI setup
if (ENABLE_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    message(STATUS "MPI found")
  else()
    message(WARNING "MPI not found — MPI backend disabled")
    set(ENABLE_MPI OFF)
  endif()
endif()

# CUDA setup
if (ENABLE_CUDA)
  cmake_policy(SET CMP0146 NEW)
  find_package(CUDAToolkit)

  if (CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found at ${CUDAToolkit_INCLUDE_DIRS}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    message(WARNING "CUDA not found — CUDA backend disabled")
    set(ENABLE_CUDA OFF)
  endif()
endif()

# OpenMP setup
if (ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
  else()
    message(WARNING "OpenMP not found — OpenMP backend disabled")
    set(ENABLE_OPENMP OFF)
  endif()
endif()

# core src
set(SRC_CORE
  src/matrix.cpp
  src/factory.cpp
)

# backend srcs
set(SRC_BACKENDS
  src/backends/cpu_backend.cpp
)

if (ENABLE_MPI)
  list(APPEND SRC_BACKENDS src/backends/mpi_backend.cpp)
endif()

if (ENABLE_CUDA)
  list(APPEND SRC_BACKENDS src/backends/cuda_backend.cu)
endif()

if (ENABLE_OPENMP)
  list(APPEND SRC_BACKENDS src/backends/omp_backend.cpp)
endif()

# build library
add_library(lumin STATIC
  ${SRC_CORE}
  ${SRC_BACKENDS}
)

# Enable Position Independent Code (required for shared libraries/Python modules)
# This is needed when linking the static library into a shared object
set_target_properties(lumin PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set compile definitions for the library target only
if (ENABLE_MPI AND MPI_FOUND)
  target_compile_definitions(lumin PRIVATE LUMIN_ENABLE_MPI)
  target_link_libraries(lumin PRIVATE MPI::MPI_CXX)
endif()

if (ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_compile_definitions(lumin PRIVATE LUMIN_ENABLE_CUDA)
  target_link_libraries(lumin PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

if (ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_compile_definitions(lumin PRIVATE LUMIN_ENABLE_OPENMP)
  target_link_libraries(lumin PRIVATE OpenMP::OpenMP_CXX)
endif()

# tests - only build if explicitly enabled (disabled by default for pip installs)
if (ENABLE_TESTS AND EXISTS ${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt)
  enable_testing()
  add_subdirectory(tests)
endif()

# Python bindings
option(ENABLE_PYTHON "Enable Python bindings" ON)
if (ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 QUIET)
  if (NOT pybind11_FOUND)
    # Try to find pybind11 via Python
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if (pybind11_DIR)
      find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
    endif()
  endif()
  
  if (pybind11_FOUND)
    message(STATUS "pybind11 found")
    
    pybind11_add_module(lumin_python python/bindings.cpp)
    target_link_libraries(lumin_python PRIVATE lumin)
    
    # Add compile definitions for Python module
    if (ENABLE_MPI AND MPI_FOUND)
      target_compile_definitions(lumin_python PRIVATE LUMIN_ENABLE_MPI)
      target_link_libraries(lumin_python PRIVATE MPI::MPI_CXX)
    endif()
    
    if (ENABLE_CUDA AND CUDAToolkit_FOUND)
      target_compile_definitions(lumin_python PRIVATE LUMIN_ENABLE_CUDA)
      target_link_libraries(lumin_python PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif()
    
    if (ENABLE_OPENMP AND OpenMP_CXX_FOUND)
      target_compile_definitions(lumin_python PRIVATE LUMIN_ENABLE_OPENMP)
      target_link_libraries(lumin_python PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Set output name to just 'lumin' for Python import
    set_target_properties(lumin_python PROPERTIES OUTPUT_NAME "lumin")
    
    # Install directly into the package directory (not a subdirectory)
    install(TARGETS lumin_python
            LIBRARY DESTINATION .
            ARCHIVE DESTINATION .
            RUNTIME DESTINATION .)

    message(STATUS "Python bindings enabled")
  else()
    message(WARNING "pybind11 not found — Python bindings disabled")
    message(STATUS "  Install pybind11: pip install pybind11")
    message(STATUS "  Or set pybind11_DIR to point to pybind11Config.cmake")
  endif()
endif()
