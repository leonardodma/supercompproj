#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace std;

struct saxpy
{
    int a;
    saxpy(int a_) : a(a_){};
    __host__ __device__ double operator()(const int &x, const int &y)
    {
        return a * x + y;
    }
};

__host__ static __inline__ int rand_10()
{
    return ((int)rand() / (RAND_MAX / 10));
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cerr << "***Numero incorreto de argumentos ***\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    // gerar numeros aleatorios
    thrust::host_vector<int> a(n);
    thrust::host_vector<int> b(n);
    thrust::host_vector<int> c(n);
    thrust::generate(a.begin(), a.end(), rand_10);
    thrust::generate(b.begin(), b.end(), rand_10);

    // transferimos para a GPU
    thrust::device_vector<int> d_a = a;
    thrust::device_vector<int> d_b = b;

    // transformacao

    thrust::transform(d_a.begin(), d_a.end(),
                      d_b.begin(), d_b.begin(),
                      saxpy(m));

    thrust::copy(d_b.begin(), d_b.end(),
                 c.begin());

    for (int i = 0; i < n; i++)
        cout << setw(6) << c[i] << " = "
             << setw(2) << m
             << "*" << setw(5) << a[i]
             << "+" << setw(5) << b[i]
             << endl;
}