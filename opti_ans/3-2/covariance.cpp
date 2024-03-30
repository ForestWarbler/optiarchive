#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

using namespace std;
typedef chrono::high_resolution_clock Clock;

#define SCALAR_VAL(x) x
/* Array initialization. */
static void init_array(int m, int n, double *&float_n, double **data) {
  int i, j;

  *float_n = (double)n;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] = ((double)i * j) / m;
}

/* Read the case file */
static void read_case(string casefile, int &n, int &m, double &float_n,
                      double **&data, double **&cov, double *&mean) {
  ifstream fin;
  fin.open(casefile.c_str());
  if (!fin.is_open()) {
    cout << "Error opening file for case" << endl;
    exit(1);
  }
  fin >> n >> m;
  float_n = (double)n;
  data = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    data[i] = (double *)malloc(m * sizeof(double));
  }
  cov = (double **)malloc(m * sizeof(double *));
  for (int i = 0; i < m; i++) {
    cov[i] = (double *)malloc(m * sizeof(double));
  }
  mean = (double *)malloc(m * sizeof(double));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      fin >> data[i][j];
  fin.close();
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, double **cov, string outfile)

{
  int i, j;
  ofstream fout;
  fout.open(outfile.c_str());
  if (!fout.is_open()) {
    cout << "Error opening file for result" << endl;
    exit(1);
  }
  fout << setprecision(10);
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      fout << cov[i][j] << " ";
    }
    fout << endl;
  }
  fout.close();
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_covariance(int n, int m, double &float_n, double **&data,
                              double **&cov, double *&mean) {
  int i, j, k;

  for (j = 0; j < m; j++) {
    mean[j] = SCALAR_VAL(0.0);
  }

  for (i = 0; i < n; i++) {
#pragma omp parallel for shared(mean)
    for (j = 0; j < m; j++) {
      mean[j] += data[i][j];
    }
  }

  for (j = 0; j < m; j++) {
    mean[j] /= float_n;
  }
#pragma omp parallel for collapse(2)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < m; i++) {
    for (j = i; j < m; j++) {
      cov[i][j] = SCALAR_VAL(0.0);
    }
  }

  for (k = 0; k < n; k++) {
#pragma omp parallel for
    for (i = 0; i < m; i++) {
      for (j = i; j < m; j++) {
        cov[i][j] += data[k][i] * data[k][j];
      }
    }
  }

  for (i = 0; i < m; i++) {
#pragma omp parallel for
    for (j = i; j < m; j++) {
      cov[i][j] /= (float_n - SCALAR_VAL(1.0));
      cov[j][i] = cov[i][j];
    }
  }
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = 0;
  int m = 0;

  /* Variable declaration/allocation. */
  double float_n;
  double **data = NULL;
  double **cov = NULL;
  double *mean = NULL;

  /* Read the option */
  int opt = 0;
  int opt_num = 0;
  string casefile = "case1.dat";
  string outfile = "result.dat";
  while ((opt = getopt(argc, argv, "hc:o:")) != -1) {
    switch (opt) {
    case 'h':
      printf("Usage: ./mycov -c <casefile> -o <outfile>\n");
      return 0;
    case 'c':
      casefile = optarg;
      opt_num += 1;
      break;
    case 'o':
      outfile = optarg;
      opt_num += 1;
      break;
    }
  }
  if (opt_num != 2) {
    printf("Usage: ./mycov -c <casefile> -o <outfile>\nError: Check your "
           "input\n");
    return 0;
  }
  read_case(casefile, n, m, float_n, data, cov, mean);

  /***************************
   * Timing starts here
   * ************************/
  cout << "Start computing..." << endl;
  auto startTime = Clock::now();

  /* Run kernel. */
  kernel_covariance(n, m, float_n, data, cov, mean);

  auto endTime = Clock::now();
  /***************************
   * Timing ends here
   * ************************/
  auto compTime =
      chrono::duration_cast<chrono::microseconds>(endTime - startTime);
  cout << "End computing..." << endl;
  cout << "Compute time=  " << compTime.count() << " microseconds" << endl;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  print_array(m, cov, outfile);

  /* Be clean. */
  for (int i = 0; i < n; i++) {
    free(data[i]);
  }
  free(data);
  for (int i = 0; i < m; i++) {
    free(cov[i]);
  }
  free(cov);
  free(mean);

  return 0;
}
