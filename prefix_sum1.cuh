#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include "wtime.h"
#include <assert.h>
#include <math.h>

using namespace std;
const int ShareMemory = 512;
//typedef int param_t;
//typedef long int index_t;
template<typename param_t>
//Belloch algorithm's downsweep step
__global__ void InterBlockSum(param_t* g_odata, param_t* BlockPreSum, const param_t n)
{
    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ int temp[ShareMemory];
    int arrayIndex=thid%blockDim.x;    
    temp[thid%blockDim.x] = g_odata[thid];
    if((thid+1)%blockDim.x==0)
    {
        temp[thid%blockDim.x] = BlockPreSum[blockIdx.x];
    }
    __syncthreads();
    
    for(int gap=n/2;gap>0;gap/=2)
    {
        if ((arrayIndex+1)%(gap*2)==0)
        {
            int tmp = temp[arrayIndex - gap];
            temp[arrayIndex - gap] = temp[arrayIndex];
            temp[arrayIndex] += tmp;
        }
        __syncthreads();
        g_odata[thid] = temp[arrayIndex];
    }
}
template<typename param_t>
__global__ void IntraBlockSum(param_t* g_idata, param_t* g_odata, param_t* BlockSum, param_t n)
{
    __shared__ int temp[ShareMemory];
    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int arrayIndex=thid%blockDim.x;

    temp[thid%blockDim.x] = g_idata[thid];
    __syncthreads();
    for(int gap=1; gap <n; gap<<=1)
    {
        if(((arrayIndex+1)%(gap*2)==0))
            temp[arrayIndex] += temp[arrayIndex-gap];
        __syncthreads();
    }
    g_odata[thid]=temp[arrayIndex];
    if((thid+1)%blockDim.x==0)
        BlockSum[blockIdx.x]=temp[arrayIndex];
}

template<typename param_t>
void my_prefixsum(param_t* A, param_t* A_d, param_t blockdim, param_t griddim, param_t* B, param_t* B_d, double& time)
{
        //typedef int data_t;
        //typedef long int index_t;
        param_t N = blockdim * griddim;//const int N = blockdim * griddim;
        param_t* B_s=(param_t*)malloc((sizeof(param_t))*N);//int* B_s=(int*)malloc((sizeof(int))*N);
        param_t size = griddim;//int size = griddim;
        param_t* BlockSum=(param_t*)malloc((sizeof(param_t))*size);//int* BlockSum=(int*)malloc((sizeof(int))*size);
        param_t* BlockPrefixSum=(param_t*)malloc((sizeof(param_t)*size));//int* BlockPrefixSum=(int*)malloc((sizeof(int)*size));
        param_t* BlockSum_d;//int* BlockSum_d;
        cudaMalloc((void**) &BlockSum_d,sizeof(param_t)*size);//cudaMalloc((void**) &BlockSum_d,sizeof(int)*size);
        param_t* BlockPrefixSum_d;//int* BlockPrefixSum_d;
        cudaMalloc((void**) &BlockPrefixSum_d,sizeof(param_t)*size);//cudaMalloc((void**) &BlockPrefixSum_d,sizeof(int)*size);

        for (int i=0;i<N;i++)
        {
            B[i]=0;
        }
        
        cudaMemcpy(B_d,B,sizeof(param_t)*N,cudaMemcpyHostToDevice);//cudaMemcpy(B_d,B,sizeof(int)*N,cudaMemcpyHostToDevice);
        double tmp1 = wtime();
        IntraBlockSum<param_t><<<griddim,blockdim>>>(A_d,B_d,BlockSum_d,N);
        //cudaThreadSynchronize();
        cudaDeviceSynchronize();
        double time2_a = wtime() - tmp1;
        cudaMemcpy(BlockSum,BlockSum_d,sizeof(param_t)*size,cudaMemcpyDeviceToHost);//cudaMemcpy(BlockSum,BlockSum_d,sizeof(int)*size,cudaMemcpyDeviceToHost); 
        cudaMemcpy(B,B_d,sizeof(param_t)*N,cudaMemcpyDeviceToHost);//cudaMemcpy(B,B_d,sizeof(long int)*N,cudaMemcpyDeviceToHost);
        
        int sumBlock=0;
        //std::cout<<"\nCPU prefix block sum:\n";
        BlockPrefixSum[0] = 0;
        double tmp2 = wtime();
        
        for (int i=0;i<size;i++)
        {
            sumBlock+=BlockSum[i];
            BlockPrefixSum[i+1]=sumBlock;
        }
        
        double time2_b = wtime() - tmp2;
        cudaMemcpy(BlockPrefixSum_d,BlockPrefixSum,sizeof(param_t)*size,cudaMemcpyHostToDevice);//cudaMemcpy(BlockPrefixSum_d,BlockPrefixSum,sizeof(int)*size,cudaMemcpyHostToDevice);
        
        double tmp3 = wtime();
        InterBlockSum<param_t><<<griddim,blockdim>>>(B_d,BlockPrefixSum_d,N);
        cudaDeviceSynchronize();
        double time2_c = wtime() - tmp3;
        time += (time2_a + time2_b + time2_c);
        cudaMemcpy(B,B_d,sizeof(param_t)*N,cudaMemcpyDeviceToHost);//cudaMemcpy(B,B_d,sizeof(long int)*N,cudaMemcpyDeviceToHost);
}

