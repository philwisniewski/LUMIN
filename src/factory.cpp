#include "lumin/factory.hpp"
#include "lumin/backend.hpp"

#include "lumin/cpu_backend.hpp"
#ifdef LUMIN_ENABLE_MPI
#include "lumin/mpi_backend.hpp"
#endif
#ifdef LUMIN_ENABLE_CUDA
#include "lumin/cuda_backend.hpp"
#endif

#include <memory>
#include <mutex>

namespace lumin {

static std::shared_ptr<Backend> default_backend_instance = nullptr;
static std::mutex backend_mutex;

std::shared_ptr<Backend> create_cpu_backend() {
  return std::make_shared<CPUBackend>();
}

#ifdef LUMIN_ENABLE_MPI
std::shared_ptr<Backend> create_mpi_backend(MPI_Comm comm) {
  return std::make_shared<MPIBackend>(comm);
}
#endif

#ifdef LUMIN_ENABLE_CUDA
std::shared_ptr<Backend> create_cuda_backend() {
  return std::make_shared<CUDABackend>();
}
#endif

void set_default_backend(std::shared_ptr<Backend> b) {
  std::lock_guard<std::mutex> lock(backend_mutex);
  default_backend_instance = std::move(b);
}

std::shared_ptr<Backend> get_default_backend() {
  std::lock_guard<std::mutex> lock(backend_mutex);
  if (!default_backend_instance) {
    default_backend_instance = std::make_shared<CPUBackend>();
  }
  return default_backend_instance;
}

}
