cmake_minimum_required(VERSION 3.16)

enable_testing()

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

# CPU tests - always built
add_executable(test_cpu test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE
    lumin
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
# Link CUDA libraries if the library was built with CUDA support
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_link_libraries(test_cpu PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()
gtest_discover_tests(test_cpu)

# MPI tests - conditionally built
if(ENABLE_MPI AND MPI_FOUND)
  add_executable(test_mpi test_mpi.cpp)
  target_compile_definitions(test_mpi PRIVATE LUMIN_ENABLE_MPI)
  target_link_libraries(test_mpi PRIVATE
      lumin
      GTest::gtest
      GTest::gtest_main
      Threads::Threads
      MPI::MPI_CXX
  )
  # Link CUDA libraries if the library was built with CUDA support
  if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(test_mpi PRIVATE CUDA::cudart CUDA::cuda_driver)
  endif()
  
  # Find MPI launcher
  find_program(MPIEXEC_EXECUTABLE
      NAMES mpiexec mpirun
      DOC "Executable for running MPI programs.")
  
  if(MPIEXEC_EXECUTABLE)
    # Use TEST_LAUNCHER property (CMake 3.18+) to wrap all tests with mpiexec
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
      set_target_properties(test_mpi PROPERTIES
          TEST_LAUNCHER ${MPIEXEC_EXECUTABLE}
      )
      # Set default number of processes for MPI tests
      set_target_properties(test_mpi PROPERTIES
          TEST_LAUNCHER_ARGS "-n;2"
      )
    endif()
  endif()
  
  # Discover tests - they will be wrapped with mpiexec if TEST_LAUNCHER is set
  gtest_discover_tests(test_mpi)
  
  if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "mpiexec not found - MPI tests will need to be run manually: mpiexec -n 2 test_mpi")
  endif()
endif()

# CUDA tests - conditionally built
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_executable(test_cuda test_cuda.cpp)
  target_compile_definitions(test_cuda PRIVATE LUMIN_ENABLE_CUDA)
  target_link_libraries(test_cuda PRIVATE
      lumin
      GTest::gtest
      GTest::gtest_main
      Threads::Threads
      CUDA::cudart
      CUDA::cuda_driver
  )
  # Set CUDA language properties if needed
  if(CMAKE_CUDA_COMPILER)
    set_target_properties(test_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
  endif()
  gtest_discover_tests(test_cuda)
endif()

# OpenMP tests - ready for when OpenMP backend is implemented
# Uncomment and adjust when ENABLE_OPENMP option is added to main CMakeLists.txt
# if(ENABLE_OPENMP)
#   find_package(OpenMP)
#   if(OpenMP_CXX_FOUND)
#     add_executable(test_openmp test_openmp.cpp)
#     target_link_libraries(test_openmp
#         lumin
#         GTest::gtest
#         GTest::gtest_main
#         Threads::Threads
#         OpenMP::OpenMP_CXX
#     )
#     gtest_discover_tests(test_openmp)
#   endif()
# endif()

