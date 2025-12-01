#ifndef MATRIX_H
#define MATRIX_H

#define MASTER 0

typedef struct {
  int rows;
  int cols;
  double *values;
} matrix;

matrix *create_matrix(int rows, int cols);
void free_matrix(matrix *mat);
void print_matrix(matrix *mat);
void generate_matrix(matrix *mat);

matrix *matrix_multiply_distributed(matrix *A, matrix *B, matrix *C, int rank, int size);
matrix *matrix_add_distributed(matrix *A, matrix *B, matrix *C, int rank, int size);
matrix *matrix_scaler_multiply(double scalar, matrix *mat, matrix *res);
matrix *matrix_scaler_multiply_distributed(double scalar, matrix *mat, matrix *res, int rank, int size);
matrix *matrix_subtract_distributed(matrix *A, matrix *B, matrix *C, int rank, int size);
matrix *matrix_transpose(matrix *mat, matrix *res);

#endif