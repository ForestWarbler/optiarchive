#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <mpi.h>

using namespace std;
typedef chrono::high_resolution_clock Clock;

#define SCALAR_VAL(x) x

/* Array initialization. */
static void init_array(int n, double**& A, double**& B)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((double)i * (j + 2) + 2) / n;
            B[i][j] = ((double)i * (j + 3) + 3) / n;
        }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
    double**& A,
    string outfile)

{
    int i, j;
    ofstream fout;
    fout.open(outfile.c_str());
    if (!fout.is_open())
    {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fout << A[i][j] << " ";
        }
        fout << endl;
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int n,
    int tsteps,
    double**& A,
    double**& B)
{
  int t, i, j;
  MPI_Status status;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double *C = new double[n * n / 2];
  double *D = new double[n * n / 2];

  for (t = 0; t < tsteps; t++) {
    if (rank == 0) {
#pragma omp parallel for simd
      for (i = 1; i < n / 2; i++) {
        j = 1;
        for (; j < n - 1 - 8; j += 8) {
          __m512d va1, va2, va3, va4, va5;
          __m512d zerop2, vresult;
          zerop2 = _mm512_set1_pd((double)0.2);
          va1 = _mm512_load_pd(&A[i][j]);
          va2 = _mm512_load_pd(&A[i][j - 1]);
          va3 = _mm512_load_pd(&A[i][j + 1]);
          va4 = _mm512_load_pd(&A[i + 1][j]);
          va5 = _mm512_load_pd(&A[i - 1][j]);
          vresult = _mm512_mul_pd(
              zerop2,
              _mm512_add_pd(_mm512_add_pd(va1, va2),
                            _mm512_add_pd(va3, _mm512_add_pd(va4, va5))));
          _mm512_store_pd(&B[i][j], vresult);
        }
        for (; j < n - 1; j++) {
          B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] +
                                       A[1 + i][j] + A[i - 1][j]);
        }
      }
    } else {
#pragma omp parallel for simd
      for (i = n / 2; i < n - 1; i++) {
        j = 1;
        for (; j < n - 1 - 8; j += 8) {
          __m512d va1, va2, va3, va4, va5;
          __m512d zerop2, vresult;
          zerop2 = _mm512_set1_pd((double)0.2);
          va1 = _mm512_load_pd(&A[i][j]);
          va2 = _mm512_load_pd(&A[i][j - 1]);
          va3 = _mm512_load_pd(&A[i][j + 1]);
          va4 = _mm512_load_pd(&A[i + 1][j]);
          va5 = _mm512_load_pd(&A[i - 1][j]);
          vresult = _mm512_mul_pd(
              zerop2,
              _mm512_add_pd(_mm512_add_pd(va1, va2),
                            _mm512_add_pd(va3, _mm512_add_pd(va4, va5))));
          _mm512_store_pd(&B[i][j], vresult);
        }
        for (; j < n - 1; j++) {
          B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] +
                                       A[1 + i][j] + A[i - 1][j]);
        }
      }

      MPI_Send(&B[n / 2][0], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
      MPI_Recv(&B[n / 2][0], n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
      MPI_Send(&B[n / 2 - 1][0], n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&B[n / 2 - 1][0], n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }

    if (rank == 0) {
#pragma omp parallel for simd
      for (i = 1; i < n / 2; i++) {
        j = 1;
        for (; j < n - 1 - 8; j += 8) {
          __m512d vb1, vb2, vb3, vb4, vb5;
          __m512d zerop2, vresult;
          zerop2 = _mm512_set1_pd((double)0.2);
          vb1 = _mm512_load_pd(&B[i][j]);
          vb2 = _mm512_load_pd(&B[i][j - 1]);
          vb3 = _mm512_load_pd(&B[i][j + 1]);
          vb4 = _mm512_load_pd(&B[i + 1][j]);
          vb5 = _mm512_load_pd(&B[i - 1][j]);
          vresult = _mm512_mul_pd(
              zerop2,
              _mm512_add_pd(_mm512_add_pd(vb1, vb2),
                            _mm512_add_pd(vb3, _mm512_add_pd(vb4, vb5))));
          _mm512_store_pd(&A[i][j], vresult);
        }
        for (; j < n - 1; j++) {
          A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] +
                                       B[1 + i][j] + B[i - 1][j]);
        }
      }
    } else {
#pragma omp parallel for simd
      for (i = n / 2; i < n - 1; i++) {
        j = 1;
        for (; j < n - 1 - 8; j += 8) {
          __m512d vb1, vb2, vb3, vb4, vb5;
          __m512d zerop2, vresult;
          zerop2 = _mm512_set1_pd((double)0.2);
          vb1 = _mm512_load_pd(&B[i][j]);
          vb2 = _mm512_load_pd(&B[i][j - 1]);
          vb3 = _mm512_load_pd(&B[i][j + 1]);
          vb4 = _mm512_load_pd(&B[i + 1][j]);
          vb5 = _mm512_load_pd(&B[i - 1][j]);
          vresult = _mm512_mul_pd(
              zerop2,
              _mm512_add_pd(_mm512_add_pd(vb1, vb2),
                            _mm512_add_pd(vb3, _mm512_add_pd(vb4, vb5))));
          _mm512_store_pd(&A[i][j], vresult);
        }
        for (; j < n - 1; j++) {
          A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] +
                                       B[1 + i][j] + B[i - 1][j]);
        }
      }
      MPI_Send(&A[n / 2][0], n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      MPI_Recv(&A[n / 2][0], n, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD, &status);
      MPI_Send(&A[n / 2 - 1][0], n, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&A[n / 2 - 1][0], n, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &status);
    }
  }
  if (rank == 1) {
    for (i = n / 2; i < n; i++) {
      for (j = 0; j < n; j++) {
        *(C + (i - n / 2) * n + j) = A[i][j];
      }
    }
    MPI_Send(C, n * n / 2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  } else if (rank == 0) {
    MPI_Recv(D, n * n / 2, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
    for (i = n / 2; i < n; i++) {
      for (j = 0; j < n; j++) {
        A[i][j] = D[(i - n / 2) * n + j];
      }
    }
  }

  delete[] C;
  delete[] D;
}

/* Read the case file */
static void read_case(string casefile, int& n, int& tsteps, double**& A, double**& B)
{
    int i, j;
    ifstream fin;
    fin.open(casefile.c_str());
    if (!fin.is_open())
    {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    fin >> n >> tsteps;

    /* Variable allocation. */
  A = (double **)aligned_alloc(64,sizeof(double *) * n);
  for (int i = 0; i < n; i++) {
    A[i] = (double *)aligned_alloc(64,sizeof(double) * n);
  }
  B = (double **)aligned_alloc(64,sizeof(double *) * n);
  for (int i = 0; i < n; i++) {
    B[i] = (double *)aligned_alloc(64,sizeof(double) * n);
  }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            fin >> A[i][j];
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            fin >> B[i][j];
    fin.close();
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = 2000;
    int tsteps = 1000;

    /* Declare the pointer to the array */
    double** A = NULL;
    double** B = NULL;

    /* Read the option */
    int opt = 0;
    int opt_num = 0;
    string casefile = "case1.dat";
    string outfile = "result.dat";
    while ((opt = getopt(argc, argv, "hc:o:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            printf("Usage: ./myjacobi -c <casefile> -o <outfile>\n");
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
    if (opt_num != 2)
    {
        printf("Usage: ./myjacobi -c <casefile> -o <outfile>\nError: Check your input\n");
        return 0;
    }
    read_case(casefile, n, tsteps, A, B);

    /***************************
     * Timing starts here
     * ************************/
    cout << "Start computing..." << endl;
    auto startTime = Clock::now();

  /* Run kernel. */

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  kernel_jacobi_2d(n, tsteps, A, B);

  /* Stop and print timer. */
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
  if (rank == 0) {
    print_array(n, A, outfile);
  }
  MPI_Finalize();
    /* Be clean. */
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
    }
    free(A);
    for (int i = 0; i < n; i++)
    {
        free(B[i]);
    }
    free(B);

    return 0;
}
