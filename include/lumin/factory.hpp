#pragma once
#include <memory>

#ifdef LUMIN_ENABLE_MPI
#include <mpi.h>
#endif

namespace lumin {

class Backend;

std::shared_ptr<Backend> create_cpu_backend();

#ifdef LUMIN_ENABLE_MPI
std::shared_ptr<Backend> create_mpi_backend(MPI_Comm comm);
#endif

#ifdef LUMIN_ENABLE_CUDA
std::shared_ptr<Backend> create_cuda_backend();
#endif

#ifdef LUMIN_ENABLE_OPENMP
std::shared_ptr<Backend> create_omp_backend();
#endif

void set_default_backend(std::shared_ptr<Backend> backend);
std::shared_ptr<Backend> get_default_backend();

}
