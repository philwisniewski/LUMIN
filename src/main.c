#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "try.h"

#define N 4 // matrix size

int main(int argc, char **argv) {
  srand(time(NULL));

  int rank, size;
  int status;

  TRY(MPI_Init(&argc, &argv), MPI_SUCCESS, "MPI_Init");
  TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS, "MPI_Comm_rank");
  TRY(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS, "MPI_Comm_size");

  matrix *A = create_matrix(4, 2); // 6x4
  matrix *B = create_matrix(2, 3); // 4x5
  matrix *C = create_matrix(4, 3); // result: 6x5
  matrix *D = create_matrix(4, 2);
  matrix *E = create_matrix(4, 2);

  if (rank == MASTER) {
    generate_matrix(A);
    generate_matrix(B);
    generate_matrix(D);

    printf("Matrix A:\n");
    print_matrix(A);
    printf("\nMatrix B:\n");
    print_matrix(B);
    printf("\nMatrix D:\n");
    print_matrix(D);
  }

  matrix_multiply_distributed(A, B, C, rank, size);
  matrix_add_distributed(A, D, E, rank, size);

  if (rank == MASTER) {
    printf("\nResult Matrix C:\n");
    print_matrix(C);
    printf("\nResult Matrix E:\n");
    print_matrix(E);
  }

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(D);
  free_matrix(E);

  TRY(MPI_Finalize(), MPI_SUCCESS, "MPI_Finalize");
  return 0;
}