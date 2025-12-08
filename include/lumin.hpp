#include "lumin/backend.hpp"
#include "lumin/cpu_backend.hpp"
#include "lumin/factory.hpp"
#include "lumin/matrix.hpp"

#ifdef LUMIN_ENABLE_CUDA
#include "lumin/cuda_backend.hpp"
#endif

#ifdef LUMIN_ENABLE_MPI
#include "lumin/mpi_backend.hpp"
#endif

#ifdef LUMIN_ENABLE_OPENMP
#include "lumin/omp_backend.hpp"
#endif

#ifdef LUMIN_ENABLE_OPENMP
#include "lumin/omp_backend.hpp"
#endif