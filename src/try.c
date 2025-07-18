#include <stdio.h>
#include "try.h"

int TRY(int exit_code, int success_code, const char * func_name) {
  if (exit_code == success_code) return 0;
  fprintf(stderr, "ERROR: Error calling %s\n", func_name);
  return exit_code;
}