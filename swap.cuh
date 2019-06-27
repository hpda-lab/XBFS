#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
//#include <string.h>
#include <utility>
using namespace std;

void swap(int* &a, int* &b){
   //printf("%d ", a[2]);
   int* temp = a;
   a = b;
   b = temp;
}
/*int main(){
    int* a = new int[5];
    std::fill(a, a+5, 3);
    //memset(a, 1, 5*sizeof(int));
    int* b = new int[5];
    std::fill(b, b+5, 2);
    //memset(b, 2, 5*sizeof(int));
    cudaSetDevice(1);
    //int* p = a;
    //a = b;
    //b = p;
    //for(int i = 0; i < 5; i++)
    //    std::cout<<a[i]<<" ";
    int* A; int* B;
    int* c = new int[5];
    std::fill(c, c+5, 10);
    int* d = new int[5];
    std::fill(d, d+5, 100);
    d[0] = 20;
    cudaMalloc((void **)&A, sizeof(int)*(5));
    cudaMemcpy(A, c, sizeof(int)*5, cudaMemcpyHostToDevice);
    //cudaMemset(A, 4, 5*sizeof(int));
    cudaMalloc((void **)&B, sizeof(int)*(5));
    cudaMemcpy(B, d, sizeof(int)*5, cudaMemcpyHostToDevice);
    //cudaMemset(B, 2, 5*sizeof(int));
    swap(A, B);
    cudaMemcpy(a, A, sizeof(int)*5, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++)
        std::cout<<a[i]<<" ";
    for(int i = 0; i < 5; i++)
        std::cout<<b[i]<<" ";
    return 0;
}*/
