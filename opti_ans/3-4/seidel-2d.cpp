#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <mutex>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <unistd.h>
#include <vector>
//#include <immintrin.h>

using namespace std;
typedef chrono::high_resolution_clock Clock;

#define SCALAR_VAL(x) x


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
static void kernel_seidel_2d(int n,
    int tsteps,
    double**& A)
{
  int t, i, j;
  int x, y;
  for (t = 0; t <= tsteps - 1; t++) {
    for (x = 0; x < 3 * n - 8; x++) {
      for (y = 1 + (x % 2); y <= min(n - 2, x + 1); y += 2) {
        j = y;
        if ((3 + x - j) / 2 <= n - 2) {
          i = (3 + x - j) / 2;
          A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
                     A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] +
                     A[i + 1][j] + A[i + 1][j + 1]) /
                    9.0;
        }
      }
    }
  }
}


/* Read the case file */
static void read_case(string casefile, int& n, int& tsteps, double**& A)
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
    A = (double**)malloc(sizeof(double*) * n);
    for (int i = 0; i < n; i++)
    {
        A[i] = (double*)malloc(sizeof(double) * n);
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            fin >> A[i][j];
    fin.close();
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = 2000;
    int tsteps = 500;

    /* Variable declaration/allocation. */
    double** A = NULL;

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
    read_case(casefile, n, tsteps, A);

    /***************************
     * Timing starts here
     * ************************/
    cout << "Start computing..." << endl;
    auto startTime = Clock::now();

    /* Run kernel. */
    kernel_seidel_2d(n, tsteps, A);

    auto endTime = Clock::now();
    /***************************
     * Timing ends here
     * ************************/
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "End computing..." << endl;
    cout << "Compute time=  " << compTime.count() << " microseconds" << endl;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    print_array(n, A, outfile);

    /* Be clean. */
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
    }
    free(A);

    return 0;
}
