cmake_minimum_required(VERSION 3.16)
project(LUMIN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options for backends
option(ENABLE_MPI "Enable MPI backend" ON)
option(ENABLE_CUDA "Enable CUDA backend" ON)

# include
include_directories(${PROJECT_SOURCE_DIR}/include)

# MPI setup
if (ENABLE_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    message(STATUS "MPI found")
    add_definitions(-DLUMIN_ENABLE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
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
    add_definitions(-DLUMIN_ENABLE_CUDA)
  else()
    message(WARNING "CUDA not found — CUDA backend disabled")
    set(ENABLE_CUDA OFF)
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

# build library
add_library(lumin STATIC
  ${SRC_CORE}
  ${SRC_BACKENDS}
)

# MPI linking
if (ENABLE_MPI)
  target_link_libraries(lumin PRIVATE MPI::MPI_CXX)
endif()

# cuda linking
if (ENABLE_CUDA)
  target_link_libraries(lumin PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

# tests ?
# if (EXISTS ${PROJECT_SOURCE_DIR}/tests)
#   enable_testing()
# endif()
