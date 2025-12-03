#pragma once
#include <memory>
#include <mpi.h>

namespace lumin {

class Backend;

std::shared_ptr<Backend> create_cpu_backend();
std::shared_ptr<Backend> create_mpi_backend(MPI_Comm comm);

void set_default_backend(std::shared_ptr<Backend> backend);
std::shared_ptr<Backend> get_default_backend();

}
