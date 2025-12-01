#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "matrix.h"
#include "try.h"

/*
 * allocate memory for matrix
 */
matrix *create_matrix(int rows, int cols) {
  matrix *mat = (matrix *) calloc(1, sizeof(matrix));
  mat->rows = rows;
  mat->cols = cols;

  // allocate memory for 2d array
  mat->values = (double *) calloc(rows * cols, sizeof(double *));

  return mat;
} /* create_matrix */

/*
 * frees memory used by a matrix
 */
void free_matrix(matrix *mat) {
  free(mat->values);
  free(mat);
} /* free_matrix */

/*
 * Function to generate random matrix of size rows x cols
 */
void generate_matrix(matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      // assign random values between 0 and 10
      mat->values[i * mat->cols + j] = rand() % 10;
    }
  }
} /* generate_matrix */

/*
 * prints out matrix
 */
void print_matrix(matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      printf("%.2f ", mat->values[i * mat->cols + j]);
    }
    printf("\n");
  }
} /* print_matrix */

/*
 * distributed matrix multiplication
 */
matrix *matrix_multiply_distributed(matrix *A, matrix *B, int rank, int size) {
  // assert matrix sizes are compatible
  if (A->cols != B->rows) {
    if (rank == MASTER) {
      fprintf(stderr, "Error: Incompatible matrices for multiplication (%dx%d * %dx%d)", A->rows, A->cols, B->rows, B->cols);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  matrix * C;
  C = create_matrix(A->rows, B->cols);

  int local_rows = A->rows / size;
  matrix *local_A = create_matrix(local_rows, A->cols);
  matrix *local_C = create_matrix(local_rows, B->cols);

  // scatter A by rows
  TRY(MPI_Scatter(
    A->values,           // send rows of A
    local_rows * A->cols, // how many elements sending
    MPI_DOUBLE,           // var type double
    local_A->values,     // local chunk of A
    local_rows * A->cols, // how many elements receiving
    MPI_DOUBLE,           // var type double
    MASTER,               // sending node
    MPI_COMM_WORLD        // all nodes
  ), MPI_SUCCESS, "MPI_Scatter");

  // broadcast full matrix B to all processes
  TRY(MPI_Bcast(
    B->values,
    B->rows * B->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  ), MPI_SUCCESS, "MPI_Bcast");

  // local computation: row-wise multiplication
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < B->cols; j++) {
      local_C->values[i * B->cols + j] = 0;
      for (int k = 0; k < A->cols; k++) {
        local_C->values[i * B->cols + j] += local_A->values[i * A->cols + k] * B->values[k * B->cols + j];
      }
    }
  }

  TRY(MPI_Gather(
    local_C->values,
    local_rows * B->cols,
    MPI_DOUBLE,
    C->values,
    local_rows * B->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  ), MPI_SUCCESS, "MPI_Gather");

  free_matrix(local_A);
  free_matrix(local_C);

  return C;
} /* matrix_multiply_distributed */


/*
 *
 */
matrix *matrix_add_distributed(matrix *A, matrix *B, int rank, int size) {
  // assert matrix sizes are compatible
  if (A->rows != B->rows || A->cols != B->cols) {
    if (rank == MASTER) {
      fprintf(stderr, "Error: Incompatible matrices for addition (%dx%d * %dx%d)", A->rows, A->cols, B->rows, B->cols);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  matrix * C;
  C = create_matrix(A->rows, A->cols);

  int local_rows = A->rows / size;
  matrix *local_A = create_matrix(local_rows, A->cols);
  matrix *local_B = create_matrix(local_rows, B->cols);
  matrix *local_C = create_matrix(local_rows, B->cols);

  MPI_Scatter(
    A->values,
    local_rows * A->cols,
    MPI_DOUBLE,
    local_A,
    local_rows * A->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  );
  MPI_Scatter(
    B->values,
    local_rows * B->cols,
    MPI_DOUBLE,
    local_B,
    local_rows * B->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  );

  for (int i = 0; i < local_rows * A->cols; i++) {
    local_C[i] = local_A[i] + local_B[i];
  }

  TRY(MPI_Gather(
    local_C->values,
    local_rows * B->cols,
    MPI_DOUBLE,
    C->values,
    local_rows * B->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  ), MPI_SUCCESS, "MPI_Gather");

  free_matrix(local_A);
  free_matrix(local_B);
  free_matrix(local_C);

  return C
}


/*
 * 
 */
matrix *matrix_scaler_multiply(double scalar, matrix *mat) {
  matrix * res = create_matrix(mat->rows, mat->cols);

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      res->values[i * res->cols + j] = mat[i * mat->rows + j] * scalar;
    }
  }

  return res;
}


/*
 *
 */
matrix *matrix_scaler_multiply_distributed(double scalar, matrix *mat, int rank, int size) {
  matrix * res;
  res = create_matrix(mat->rows, mat->cols);

  int local_rows = mat->rows / size;
  matrix *local_mat = create_matrix(local_rows, mat->cols);

  TRY(MPI_Scatter(
    mat->values,
    local_rows * mat->cols,
    MPI_DOUBLE,
    local_mat,
    local_rows * mat->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  ), MPI_SUCCESS, "MPI_Scatter");

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      local_mat->values[i * mat->cols + j] *= scalar;
    }
  }

  TRY(MPI_Gather(
    local_mat->values,
    local_rows * local_mat->cols,
    MPI_DOUBLE,
    res->values,
    local_rows * local_mat->cols,
    MPI_DOUBLE,
    MASTER,
    MPI_COMM_WORLD
  ), MPI_SUCCESS, "MPI_Gather");

  free_matrix(local_mat);

  return res;
}


matrix *matrix_subtract_distributed(matrix *A, matrix *B, matrix *C, int rank, int size);
matrix *matrix_transpose(matrix *mat, matrix *res);
