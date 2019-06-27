#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

static void HandleError( cudaError_t err, const char *file, int line ) {
       if (err != cudaSuccess) {
           printf( "\n%s in %s at line %d\n", \
           cudaGetErrorString( err ), file, line );
           exit( EXIT_FAILURE );
        }
      }
#define H_ERR( err ) \
      (HandleError( err, __FILE__, __LINE__ ))

using namespace std;
int cmp(const void* a, const void* b){
    return *(int*)a - *(int*)b;
}

template<typename param_t, typename index_t>
__global__ void atomic_FQ_Generator2(param_t* SA, param_t* small_FQ, param_t* medium_FQ, param_t* large_FQ, param_t* small, param_t* medium, param_t* large, param_t vert_count, param_t level, index_t* beg_cu){
    param_t tid = blockDim.x*blockIdx.x+threadIdx.x;
	param_t m1 = 0, m2 = 0, m3 = 0;
    param_t var;
    while(tid < vert_count){
        var = SA[tid];
        if(var==level){
          if(beg_cu[tid+1]-beg_cu[tid]<=32){
             m1 = atomicAdd(small, 1);
		     small_FQ[m1] = tid;
          }
          else if((beg_cu[tid+1] - beg_cu[tid])>32 && (beg_cu[tid+1] - beg_cu[tid])<=32){
             m2 = atomicAdd(medium, 1);
             medium_FQ[m2] = tid;
          }
          else if(beg_cu[tid+1]-beg_cu[tid]>32){
             m3 = atomicAdd(large, 1);
             large_FQ[m3] = tid;
          }
       }
	   tid += blockDim.x * gridDim.x;
    }
    //__syncthreads();   
}

template<typename param_t, typename index_t>
__global__ void small_SA_TD_update2(param_t* SA, param_t* small_FQ, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    //param_t count= (small[0] < vert_count) ? small[0] : vert_count;
    while(tid < small[0]){  
        param_t front_vert = small_FQ[tid];
        if(SA[front_vert]==level){
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            param_t front = csr_cu[my_beg];
            param_t SA_status = SA[front];
            if(SA_status==-1){  //unvisited neighbor
                SA[front] = level+1;  //mark update the status array!
            }
            my_beg++;
        }}
        tid += blockDim.x * gridDim.x;
    }
    //__syncthreads();
}
template<typename param_t, typename index_t>
__global__ void small_SA_TD_update3(param_t* SA, param_t* small_FQ, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < small[0]){
        param_t front_vert = small_FQ[tid];
        if(SA[front_vert]==level){
        
        }
    }
}
template<typename param_t, typename index_t>
__global__ void medium_SA_TD_update2(param_t* SA, param_t* medium_FQ, param_t* medium, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t wid = tid >> 5;
    //param_t count= (medium[0] < vert_count) ? medium[0] : vert_count;
    const param_t woff = tid & 31;
    while(wid < medium[0]){
        param_t front_vert = medium_FQ[wid];
        if(SA[front_vert]==level){
        index_t my_beg = beg_cu[medium_FQ[wid]] + woff;//tid%32;
        index_t my_end = beg_cu[medium_FQ[wid]+1];
        while(my_beg < my_end){
            param_t vertex = csr_cu[my_beg];
            param_t SA_status = SA[vertex];
            if(SA_status==-1)
                SA[vertex] = level+1;
            my_beg += 32; 
        }}
        wid += blockDim.x * gridDim.x / 32;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void large_SA_TD_update2(param_t* SA, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;    
    //param_t count= (large[0] < vert_count) ? large[0] : vert_count;
    param_t bid = tid / blockDim.x;
    while(bid < large[0]){
        param_t front_vert = large_FQ[bid];
        if(SA[front_vert]==level){
        index_t my_beg = beg_cu[large_FQ[bid]] + threadIdx.x;//tid%256;
        index_t my_end = beg_cu[large_FQ[bid]+1];
        while(my_beg < my_end){
            param_t vertex = csr_cu[my_beg];
            param_t SA_status = SA[vertex];
            if(SA_status==-1)
                SA[vertex] = level+1;
            my_beg += blockDim.x;
        }}   
        bid += gridDim.x;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void BU_FQ_find2(param_t* SA, param_t* small_BU, param_t* medium_BU, param_t* large_BU, param_t* small, param_t* medium, param_t* large, index_t* beg_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t m1 = 0, m2 = 0, m3 = 0;
    param_t status = -1;
    while(tid < vert_count){
        status = SA[tid];
        if(status==-1){ //unvisited
           if(beg_cu[tid+1]-beg_cu[tid]>0 && beg_cu[tid+1]-beg_cu[tid]<=32){
               m1 = atomicAdd(small, 1);
               small_BU[m1] = tid;
           }
           else if((beg_cu[tid+1] - beg_cu[tid])>32 && (beg_cu[tid+1] - beg_cu[tid])<=256){
               m2 = atomicAdd(medium, 1);
               medium_BU[m2] = tid;
           }
           else if(beg_cu[tid+1]-beg_cu[tid]>256){
               m3 = atomicAdd(large, 1);
               large_BU[m3] = tid;
           }
        }
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void small_SA_BU_update2(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < small[0]){    
        //if(small_BU[tid]!=-1){
        param_t front_vert = small_BU[tid];
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            param_t status = SA[csr_cu[my_beg]];
            if(status==level){
                SA[front_vert] = level+1;
                break;
            }
            my_beg++;
        }
        //}
        tid += blockDim.x*gridDim.x;
    }
    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void medium_SA_BU_update2(param_t* SA, param_t* medium_BU, param_t* medium, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t wid = tid >> 5;
    const param_t woff = tid & 31;
    while(wid < medium[0]){   
        //if(medium_BU[wid]!=-1){ //
        param_t front_vert = medium_BU[wid];
        index_t my_beg = beg_cu[front_vert] + woff;
        index_t my_end = beg_cu[front_vert + 1];
        while(__any_sync(0xffffffff, my_beg < my_end)){
            param_t SA_neighbor = level-1;
            if(my_beg < my_end)
            {
                param_t my_neighbor = csr_cu[my_beg];
                SA_neighbor = SA[my_neighbor];
            }
            //param_t my_neighbor = csr_cu[my_beg];
            if(__any_sync(0xffffffff, SA_neighbor==level)){//if(SA[my_neighbor]==level){
                if(woff==0)
                    SA[front_vert] = level+1;
                break;
            }
            my_beg += 32;
        }
        //}
        wid += blockDim.x * gridDim.x / 32;
    }
    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void large_SA_BU_update2(param_t* SA, param_t* large_BU, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t bid = tid / blockDim.x;
    while(bid < large[0]){
        //if(large_BU[bid]!=-1){
        param_t front_vert = large_BU[bid];
        index_t my_beg = beg_cu[large_BU[bid]] + threadIdx.x;
        index_t my_end = beg_cu[large_BU[bid]+1];
        //param_t my_neighbor = csr_cu[my_beg];
        while(__syncthreads_or(my_beg < my_end)){
            param_t SA_neighbor = level-1;
            if(my_beg < my_end)
            {
                param_t my_neighbor = csr_cu[my_beg];
                SA_neighbor = SA[my_neighbor];
            }
            if(__syncthreads_or(SA_neighbor==level)){
            //if(SA[my_neighbor]==level){
                if(threadIdx.x==0)
                   SA[front_vert] = level+1;
                break;
            }
            my_beg += blockDim.x;
        }
        //}
        bid += gridDim.x;
    }
    __syncthreads();
}


