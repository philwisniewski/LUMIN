#include "lumin/mpi_backend.hpp"
#include "lumin/matrix.hpp"
#include "lumin/backend.hpp"

#include <mpi.h>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace lumin {

static void mpi_abort_print(int rank, const std::string &msg) {
  if (rank == 0) std::cerr << "MPIBackend error: " << msg << std::endl;
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void compute_counts_displs_rows(int total_rows, int cols, int world_size,
                                       std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(world_size, 0);
  displs.assign(world_size, 0);

  int base = total_rows / world_size;
  int rem = total_rows % world_size;

  for (int r = 0; r < world_size; r++) {
    int rows_for_r = base + (r < rem ? 1 : 0);
    counts[r] = rows_for_r * cols;
  }

  displs[0] = 0;
  for (int r = 1; r < world_size; ++r) {
      displs[r] = displs[r-1] + counts[r-1];
  }
}

MPIBackend::MPIBackend(MPI_Comm comm)
  : m_comm(comm)
{
  MPI_Comm_rank(m_comm, &m_rank);
  MPI_Comm_size(m_comm, &m_size);
}

Matrix MPIBackend::add(const Matrix& A, const Matrix& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    mpi_abort_print(m_rank, "add: dimension mismatch");
  }

  int total_rows = static_cast<int>(A.rows());
  int cols = static_cast<int>(A.cols());

  std::vector<int> counts, displs;
  compute_counts_displs_rows(total_rows, cols, m_size, counts, displs);

  int local_elems = counts[m_rank];
  std::vector<double> localA(local_elems);
  std::vector<double> localB(local_elems);
  std::vector<double> localC(local_elems, 0.0);

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr), // sendbuf
    counts.data(), // sendcounts
    displs.data(), // displs
    MPI_DOUBLE, // sendtype
    localA.data(), // recvbuf
    local_elems, // recvcount
    MPI_DOUBLE, // recvtype
    0, // root
    m_comm // comm
  );

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(B.data()) : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    localB.data(),
    local_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );

  // local compute
  for (int i = 0; i < local_elems; i++) {
    localC[i] = localA[i] + localB[i];
  }

  Matrix C;
  if (m_rank == 0) {
    C = Matrix(static_cast<size_t>(total_rows), static_cast<size_t>(cols));
  }

  MPI_Gatherv(
    localC.data(),
    local_elems,
    MPI_DOUBLE,
    (m_rank == 0 ? C.data() : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    0,
    m_comm
  );
  
  return (m_rank == 0) ? C : Matrix(0, 0);
}

Matrix MPIBackend::subtract(const Matrix& A, const Matrix& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    mpi_abort_print(m_rank, "subtract: dimension mismatch");
  }

  int total_rows = static_cast<int>(A.rows());
  int cols = static_cast<int>(A.cols());

  std::vector<int> counts, displs;
  compute_counts_displs_rows(total_rows, cols, m_size, counts, displs);

  int local_elems = counts[m_rank];
  std::vector<double> localA(local_elems);
  std::vector<double> localB(local_elems);
  std::vector<double> localC(local_elems, 0.0);

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr), // sendbuf
    counts.data(), // sendcounts
    displs.data(), // displs
    MPI_DOUBLE, // sendtype
    localA.data(), // recvbuf
    local_elems, // recvcount
    MPI_DOUBLE, // recvtype
    0, // root
    m_comm // comm
  );

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(B.data()) : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    localB.data(),
    local_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );

  // local compute
  for (int i = 0; i < local_elems; i++) {
    localC[i] = localA[i] - localB[i];
  }

  Matrix C;
  if (m_rank == 0) {
    C = Matrix(static_cast<size_t>(total_rows), static_cast<size_t>(cols));
  }

  MPI_Gatherv(
    localC.data(),
    local_elems,
    MPI_DOUBLE,
    (m_rank == 0 ? C.data() : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    0,
    m_comm
  );
  
  return (m_rank == 0) ? C : Matrix(0, 0);
}

Matrix MPIBackend::scalar(double s, const Matrix& A) {
  int total_rows = static_cast<int>(A.rows());
  int cols = static_cast<int>(A.cols());
  std::vector<int> counts, displs;
  compute_counts_displs_rows(total_rows, cols, m_size, counts, displs);

  int local_elems = counts[m_rank];
  std::vector<double> localA(local_elems);
  std::vector<double> localR(local_elems, 0.0);

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    localA.data(),
    local_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );

  for (int i = 0; i < local_elems; i++) {
    localR[i] = localA[i] * s;
  }

  Matrix R;
  if (m_rank == 0) {
    R = Matrix(static_cast<size_t>(total_rows), static_cast<size_t>(cols));
  }

  MPI_Gatherv(
    localR.data(),
    local_elems,
    MPI_DOUBLE,
    (m_rank == 0 ? R.data() : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    0,
    m_comm
  );

  return (m_rank == 0) ? R : Matrix(0, 0); 
}

Matrix MPIBackend::multiply(const Matrix& A, const Matrix& B) {
  if (A.cols() != B.rows()) {
    mpi_abort_print(m_rank, "multiply: incompatible matrix dimensions");
  }

  int total_rows = static_cast<int>(A.rows());
  int a_cols = static_cast<int>(A.cols());
  int b_cols = static_cast<int>(B.cols());

  std::vector<int> countsA, displsA;
  compute_counts_displs_rows(total_rows, a_cols, m_size, countsA, displsA);

  std::vector<int> countsC, displsC;
  compute_counts_displs_rows(total_rows, b_cols, m_size, countsC, displsC);

  int localA_elems = countsA[m_rank];
  int localC_elems = countsC[m_rank];
  int local_rows = (a_cols == 0) ? 0 : localA_elems / a_cols;

  std::vector<double> localA(localA_elems);
  std::vector<double> localC(localC_elems, 0.0);

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr),
    countsA.data(),
    displsA.data(),
    MPI_DOUBLE,
    (localA_elems ? localA.data() : nullptr),
    localA_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );

  std::vector<double> Bbuf;
  if (m_rank == 0) {
    Bbuf.assign(B.data(), B.data() + static_cast<size_t>(a_cols * b_cols));
  }
  else {
    Bbuf.assign(static_cast<size_t>(a_cols * b_cols), 0.0);
  }

  MPI_Bcast(Bbuf.data(), a_cols * b_cols, MPI_DOUBLE, 0, m_comm);

  for (int i = 0; i < local_rows; ++i) {
    for (int k = 0; k < a_cols; ++k) {
      double a_val = localA[i * a_cols + k];
      const double *b_row = &Bbuf[static_cast<size_t>(k) * b_cols];
      double *c_row = &localC[static_cast<size_t>(i) * b_cols];
      for (int j = 0; j < b_cols; ++j) {
        c_row[j] += a_val * b_row[j];
      }
    }
  }

  Matrix C;
  if (m_rank == 0) {
    C = Matrix(static_cast<size_t>(total_rows), static_cast<size_t>(b_cols));
  }

  MPI_Gatherv((localC_elems ? localC.data() : nullptr),
    localC_elems,
    MPI_DOUBLE,
    (m_rank == 0 ? C.data() : nullptr),
    countsC.data(),
    displsC.data(),
    MPI_DOUBLE,
    0,
    m_comm
  );

  return (m_rank == 0) ? C : Matrix(0, 0);
}

double MPIBackend::dot(const Matrix& A, const Matrix& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    mpi_abort_print(m_rank, "dot: dimension mismatch");
  }

  int total_rows = static_cast<int>(A.rows());
  int cols = static_cast<int>(A.cols());

  std::vector<int> counts, displs;
  compute_counts_displs_rows(total_rows, cols, m_size, counts, displs);

  int local_elems = counts[m_rank];
  std::vector<double> localA(local_elems);
  std::vector<double> localB(local_elems);
  double localTotal = 0.0; 

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr), // sendbuf
    counts.data(), // sendcounts
    displs.data(), // displs
    MPI_DOUBLE, // sendtype
    localA.data(), // recvbuf
    local_elems, // recvcount
    MPI_DOUBLE, // recvtype
    0, // root
    m_comm // comm
  );

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(B.data()) : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    localB.data(),
    local_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );
  
  // local compute
  for (int i = 0; i < local_elems; i++) {
    localTotal += localA[i] * localB[i];
  }

  double res = 0.0;

  MPI_Reduce(
    &localTotal,
    (m_rank == 0 ? &res : nullptr),
    1,
    MPI_DOUBLE,
    MPI_SUM,
    0,
    m_comm
  );
  
  return res;
}

Matrix MPIBackend::transpose(const Matrix& A) {
  int total_rows = static_cast<int>(A.rows());
  int cols = static_cast<int>(A.rows());
  std::vector<int> counts, displs;
  compute_counts_displs_rows(total_rows, cols, m_size, counts, displs);

  int local_elems = counts[m_rank];
  std::vector<double> localBuf(local_elems);

  MPI_Scatterv(
    (m_rank == 0 ? const_cast<double*>(A.data()) : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    (local_elems ? localBuf.data() : nullptr),
    local_elems,
    MPI_DOUBLE,
    0,
    m_comm
  );

  Matrix gathered;
  if (m_rank == 0) {
    gathered = Matrix(static_cast<size_t>(total_rows), static_cast<size_t>(cols));
  }

  MPI_Gatherv(
    (local_elems ? localBuf.data() : nullptr),
    local_elems,
    MPI_DOUBLE,
    (m_rank == 0 ? gathered.data() : nullptr),
    counts.data(),
    displs.data(),
    MPI_DOUBLE,
    0,
    m_comm
  );

  if (m_rank == 0) {
    Matrix res = Matrix(static_cast<size_t>(cols), static_cast<size_t>(total_rows));
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        res.data()[static_cast<size_t>(j) * res.cols() + i] =
        gathered.data()[static_cast<size_t>(i) * cols + j];
      }
    }
    return res;
  }

  return Matrix(0,0);
}

}
