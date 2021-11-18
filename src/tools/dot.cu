/*
完整的向量点积CUDA程序
a=[a1,a2,…an], b=[b1,b2,…bn]
a*b=a1*b1+a2*b2+…+an*bn
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda.h> 
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#define N 10
__global__ void Dot(int *a, int *b, int *c) //声明kernel函数{
    __shared__ int temp[N]; // 声明在共享存储中的变量
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    __syncthreads();  // 此处有下划线，并不是错误，而是VS智能感知有问题，
    if (0 == threadIdx.x)
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
            sum += temp[i];
        *c = sum;
        printf("sum Calculated on Device:%d\n", *c);
    }
}
void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
        *(a + i) = rand() % 10;
}
int main()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, sizeof(int));
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(sizeof(int));
    printf("Array a[N]:\n");
    for (int i = 0; i < N; i++) printf("%d ", a[i]);
    printf("\n");
    printf("Array b[N]:\n");
    for (int i = 0; i < N; i++) printf("%d ", b[i]);
    printf("\n\n");
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    Dot << <1, N >> > (d_a, d_b, d_c); //单block多thread
    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    int sumHost = 0;
    for (int i = 0; i < N; i++)
        sumHost += a[i] * b[i];
    printf("sum Calculated on Host=%d\n", sumHost);
    printf("Device to Host: a*b=%d\n", *c);
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    while(1){
        ;
    }
    return 0;
}
