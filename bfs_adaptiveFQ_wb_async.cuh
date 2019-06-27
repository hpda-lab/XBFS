#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <numeric>
#include "./bfs_single_scan.cuh"
#include "./bfs_TD_scan_free.cuh"
//#include "bfs_online_before_bottom-up.cuh"
//#include "bfs_online1.cuh"
#include "prefix_sum1.cuh"
#include "swap.cuh"
#include "workload_gap.cuh"
//#include "workload_distribution.cuh"
#include <fstream>
//#include "bfs_workload_optimize1.cuh"


//#define SML_MED_THRESH  (1<<9) //(1<<22)
//#define MED_LRG_THRESH  (1<<10)  //(1<<30)

#define SML_MED_THRESH  (1<<22)
#define MED_LRG_THRESH  (1<<30)
#define PADDING 3
#define INFINITE 10000
//#define SML_MED_THRESH   (128)
//#define MED_LRG_THRESH  (1024)



//static void HandleError( cudaError_t err, const char *file, int line ) {
//       if (err != cudaSuccess) {
//           printf( "\n%s in %s at line %d\n", \
//           cudaGetErrorString( err ), file, line );
//           exit( EXIT_FAILURE );
//        }
//      }
//#define H_ERR( err ) \
//      (HandleError( err, __FILE__, __LINE__ ))
//
//using namespace std;
int cmp1(const void* a, const void* b){
    return *(long int*)a - *(long int*)b;
}

template<typename param_t, typename index_t>
__global__ void small_SA_TD_update(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* small_FQ1, param_t* medium_FQ1, param_t* large_FQ1, param_t* small1, param_t* med1, param_t* large1){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    param_t count= (small[0] < vert_count) ? small[0] : vert_count;
    while(tid < count){  
        param_t front_vert = small_FQ[tid];
        //if(SA[front_vert]==level){
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            param_t front = csr_cu[my_beg];
            param_t next_beg = beg_cu[front];
            param_t next_end = beg_cu[front+1];
            param_t SA_status = SA[front];
            if(SA_status==-1){  //unvisited neighbor, attempting.
                //SA[front] = level+1;  
                int m = atomicCAS(&SA[front], -1, level+1);
                if(m==-1){
                    if((next_end-next_beg)>0 && (next_end-next_beg)<=32){
                        small_FQ1[atomicAdd(small1, 1)] = front;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ1[atomicAdd(med1, 1)] = front;
                    }
                    else if((next_end-next_beg)>256){
                        large_FQ1[atomicAdd(large1, 1)] = front;
                    }
                }
            }
            my_beg++;
        }//}
    tid += blockDim.x * gridDim.x;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void medium_SA_TD_update(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* small_FQ1, param_t* medium_FQ1, param_t* large_FQ1, param_t* small1, param_t* med1, param_t* large1){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t wid = tid >> 5;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    param_t count= (medium[0] < vert_count) ? medium[0] : vert_count;
    const param_t woff = tid & 31;
    while(wid < count){
        param_t front_vert = medium_FQ[wid];
        //if(SA[front_vert]==level){
        index_t my_beg = beg_cu[front_vert] + woff;//tid%32;
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            param_t vertex = csr_cu[my_beg];
            param_t next_beg = beg_cu[vertex];
            param_t next_end = beg_cu[vertex+1];
            param_t SA_status = SA[vertex];
            if(SA_status==-1){
                //SA[vertex] = level+1;
                int m = atomicCAS(&SA[vertex], -1, level+1);
                if(m==-1){
                    if((next_end-next_beg)>0 && (next_end-next_beg)<=32){
                        small_FQ1[atomicAdd(small1, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ1[atomicAdd(med1, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>256){
                        large_FQ1[atomicAdd(large1, 1)] = vertex;
                    }
                }
            }
            my_beg += 32; 
        }//}
    wid += blockDim.x * gridDim.x / 32;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void large_SA_TD_update(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* small_FQ1, param_t* medium_FQ1, param_t* large_FQ1, param_t* small1, param_t* med1, param_t* large1){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;    
    param_t bid = tid / blockDim.x;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    const param_t count= (large[0] < vert_count) ? large[0] : vert_count;
    while(bid < count){
        param_t front_vert = large_FQ[bid];
        //if(SA[front_vert]==level){
        index_t my_beg = beg_cu[front_vert] + threadIdx.x;//tid%256;
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            param_t vertex = csr_cu[my_beg];
            param_t next_beg = beg_cu[vertex];
            param_t next_end = beg_cu[vertex+1];
            param_t SA_status = SA[vertex];
            if(SA_status==-1){
                //SA[vertex] = level+1;
                int m = atomicCAS(&SA[vertex], -1, level+1);
                if(m==-1){
                    if((next_end-next_beg)>0 && (next_end-next_beg)<=32){
                        small_FQ1[atomicAdd(small1, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ1[atomicAdd(med1, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>256){
                        large_FQ1[atomicAdd(large1, 1)] = vertex;
                    }
                }
            }
            my_beg += blockDim.x;
        }//}   
        bid += gridDim.x;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void BU_FQ_find(param_t* SA, param_t* small_BU, param_t* medium_BU, param_t* large_BU, param_t* small, param_t* medium, param_t* large, index_t* beg_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t m1 = 0, m2 = 0, m3 = 0;
    param_t status = -1; 
    while(tid < vert_count){
        status = SA[tid];
        if(status==-1){ //unvisited
            if(beg_cu[tid+1]-beg_cu[tid]>0 && beg_cu[tid+1]-beg_cu[tid]<=128){
                m1 = atomicAdd(small, 1);
                small_BU[m1] = tid;
            }
            else if((beg_cu[tid+1] - beg_cu[tid])>128 && (beg_cu[tid+1] - beg_cu[tid])<=1024){
                m2 = atomicAdd(medium, 1);
                medium_BU[m2] = tid;
            }
            else if(beg_cu[tid+1]-beg_cu[tid]>1024){
                m3 = atomicAdd(large, 1);
                large_BU[m3] = tid;
            }
        }
        tid += blockDim.x * gridDim.x;
    }
    //__syncthreads();
}
template<typename param_t, typename index_t>
__global__ void ballot_counting(param_t* SA, param_t vert_count, param_t res, param_t* s_count, param_t* m_count, param_t* l_count, index_t* beg_cu){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int my_beg = tid * (vert_count/(blockDim.x*gridDim.x));
    int my_end = (tid+1) * (vert_count/(blockDim.x*gridDim.x));
    int gap = vert_count/(blockDim.x*gridDim.x);
    if(tid >= blockDim.x*gridDim.x-res){
        my_beg+=(tid - (blockDim.x*gridDim.x-res));
        my_end=my_beg+(gap+1);
    }
    int laneId = threadIdx.x & 0x1f;
    int woff = tid % 32;
    while(__any_sync(0xffffffff, my_beg < my_end)){
        unsigned my_small, my_medium, my_large;
        for(int i = 0; i < 32; i++){
            int status = 0;
            int deg = 0;
            int wbeg = __shfl_sync(0xffffffff, my_beg, i, 32) + woff;
            int wend = __shfl_sync(0xffffffff, my_end, i, 32);  
            if(wbeg < wend){
                status = SA[wbeg];
                deg = beg_cu[wbeg+1] - beg_cu[wbeg];
            }
            unsigned small_flag =  __ballot_sync(0xffffffff, status==-1);

            //unsigned small_flag =  __ballot_sync(0xffffffff, status==-1&&deg>0&&deg<=1024);
            //unsigned med_flag =  __ballot_sync(0xffffffff, status==-1&&deg>1024&&deg<=2048);
            //unsigned large_flag =  __ballot_sync(0xffffffff, status==-1&&deg>2048);

            //unsigned small_flag =  __ballot_sync(0xffffffff, status==-1&&deg>0&&deg<=SML_MED_THRESH);
            //unsigned med_flag =  __ballot_sync(0xffffffff, status==-1&&deg>SML_MED_THRESH&&deg<=MED_LRG_THRESH);
            //unsigned large_flag =  __ballot_sync(0xffffffff, status==-1&&deg>MED_LRG_THRESH);
            if(laneId==i){
                my_small = small_flag;
            }
        }
        //int iter = ((my_end - my_beg) > 32)? 32 : (my_end-my_beg);
        //for(int i = 0; i < iter; i++){
        //    if(((my_flag & (1 << i))>>i) ==1){        
        //    }
        //}
        //mycount += __popc(my_flag);
        s_count[tid] += __popc(my_small);
        my_beg+=32;
    }
}
template<typename param_t, typename index_t>
__global__ void FQ_gen(param_t* SA, param_t vert_count, param_t res, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, param_t* s_offset, param_t* m_offset, param_t* l_offset, param_t* s_count, param_t* m_count, param_t* l_count, index_t* beg_cu){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int my_beg = tid * (vert_count/(blockDim.x*gridDim.x));
    int my_end = (tid+1) * (vert_count/(blockDim.x*gridDim.x));
    int gap = vert_count/(blockDim.x*gridDim.x);
    if(tid >= blockDim.x*gridDim.x-res){
        my_beg+=(tid - (blockDim.x*gridDim.x-res));
        gap++;
        my_end=my_beg+gap;
    }
    int small_off = s_offset[tid];
    if(tid==0){
        small[0] = s_offset[blockDim.x*gridDim.x-1] + s_count[blockDim.x*gridDim.x-1];
    }
    int laneId = threadIdx.x & 0x1f;
    int woff = tid % 32;
    while(__any_sync(0xffffffff, my_beg < my_end)){
        unsigned my_small, my_medium, my_large;
        for(int i = 0; i < 32; i++){
            int status = -1;
            int deg = 0;
            int wbeg = __shfl_sync(0xffffffff, my_beg, i, 32) + woff;
            int wend = __shfl_sync(0xffffffff, my_end, i, 32);
            if(wbeg < wend){
                status = SA[wbeg];
                deg = beg_cu[wbeg+1] - beg_cu[wbeg];
            }
            unsigned small_flag =  __ballot_sync(0xffffffff, status==-1);

            //unsigned small_flag =  __ballot_sync(0xffffffff, status==-1&&deg>0&&deg<=1024);
            //unsigned med_flag =  __ballot_sync(0xffffffff, status==-1&&deg>1024&&deg<=2048);
            //unsigned large_flag =  __ballot_sync(0xffffffff, status==-1&&deg>2048);

            //unsigned small_flag =  __ballot_sync(0xffffffff, status==-1&&deg>0&&deg<=SML_MED_THRESH);
            //unsigned med_flag =  __ballot_sync(0xffffffff, status==-1&&deg>SML_MED_THRESH&&deg<=MED_LRG_THRESH);
            //unsigned large_flag =  __ballot_sync(0xffffffff, status==-1&&deg>MED_LRG_THRESH);
            if(i==laneId){
                my_small = small_flag;
            }
        }
        int iter = ((my_end - my_beg) > 32)? 32 : (my_end-my_beg);
        for(int i = 0; i < iter; i++){
            if(((my_small & (1 << i))>>i) ==1){
                small_FQ[small_off++] = my_beg+i;
            }
        }
        my_beg+=32;
    }
}

template<typename param_t, typename index_t>
__global__ void small_SA_BU_update(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, index_t* Count, index_t* deg){
    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
    param_t tid = TID;//threadIdx.x + blockIdx.x * blockDim.x;
    //int counts = 0;
    while(tid < small[0]){    
        int count=0;
        index_t counts = 0;
        bool is_curr = false;
        param_t front_vert = small_BU[tid];  //all unvisited vertices, needs to be divided into 3 types!!!
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        index_t gap = beg_cu[front_vert+1] - beg_cu[front_vert];
        while(my_beg < my_end){
            counts++;
            param_t neigh = csr_cu[my_beg];
            param_t status = SA[neigh];
            if(status==level){
                SA[front_vert] = level+1;
                count++;
                //deg[tid] = counts;
                break;
            }
            if(is_curr==false)
                is_curr = (status==level+1);
            my_beg++;
        }
        if(count==0&&is_curr==true){
            SA[front_vert] = level+2;
        }
        deg[tid] = counts;
        //Count[TID] += counts;
        tid += blockDim.x*gridDim.x;
    }
    //Count[TID] = counts;
}
template<typename param_t, typename index_t>
__global__ void small_BU_update1(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count/*, index_t* Count, index_t* deg*/, param_t* counter, param_t padding){
    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
    param_t tid = padding*TID;
    while(tid < small[0]){
        for(int i = 0; i < padding; i++){
            if(tid < small[0]){
                int count=0;
                bool is_curr = false;
                param_t front_vert = small_BU[tid];
                index_t my_beg = beg_cu[front_vert];
                index_t my_end = beg_cu[front_vert+1];
                while(my_beg < my_end){
                    param_t neigh = csr_cu[my_beg];
                    param_t status = SA[neigh];
                    if(status==level){
                        SA[front_vert] = level+1;
                        count++;
                        break;
                    }
                    if(is_curr==false)
                        is_curr = (status==level+1);
                    my_beg++;
                }
                if(count==0&&is_curr==true)
                    SA[front_vert] = level+2;
                tid++;
            }
            else{  break;  }
        }
        tid = atomicAdd(counter, padding*(counter[0]<small[0]));
    }
}

    template<typename param_t, typename index_t>
__global__ void naive_BU(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* counter, index_t* deg)
{
    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
    param_t tid = TID;
    //bool my_finish=false;
    while(tid < small[0])
    {
        int count=0;
        bool is_curr = false;
        param_t front_vert = small_BU[tid];
        //if(front_vert!=-1){
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        index_t counts=0;
        while(my_beg < my_end)
        {
            counts++;
            param_t neigh = csr_cu[my_beg];
            param_t status = SA[neigh];
            if(status==level)
            {
                SA[front_vert] = level+1;
                //    if(front_vert==2304370) printf("%d ", counts);
                count++;
                //        my_finish=true;
                break;
            }
            if(is_curr==false)
            {
                is_curr = (status==level+1);
                //    if(is_curr==true)
                //        break;
            }
            my_beg++;
        }
        //deg[tid] = counts;
        if(count==0&&is_curr==true)
        {
            //if(is_curr==true){
            SA[front_vert] = level+2;
            //    my_finish=true;
            //    break;
        }
        //tid = atomicAdd(counter, counter[0]<small[0]); //&&tid<small[0]
        tid = atomicAdd(counter, 1); //&&tid<small[0]
        //Count[TID] += counts;
        }
    }
    template<typename param_t, typename index_t>
        __global__ void intra_warp_BU(param_t* SA, param_t* FQ_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, index_t* Count, param_t* counter){
            const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
            param_t tid = TID;
            param_t wid = TID>>5;
            param_t wbeg;

            //assert(blockDim.x == 128);
            //__shared__ int overall[512];  // 4 warps in a block: blockDimx.x=128
            //__shared__ int overall[512];  // 4 warps in a block: blockDimx.x=128
            bool my_finish = false;//is_finish[tid];
            //int reverseId;
            //__shared__ int counts[128];
            //__shared__ int B[16]; //unfinished threadID array
            const int laneId = threadIdx.x & 0x1f;
            const int warpId = threadIdx.x >> 5;
            //printf("%d ", beg_cu[FQ_BU[2245035]+1]);
            //printf("%d ", beg_cu[FQ_BU[2245035]]);  //degree=32!!
            //index_t beg = beg_cu[FQ_BU[2245035]];
            //printf("%d ", SA[csr_cu[beg]]);  //4,   So FQ_BU[2245035]=3072563 should be at level 5!!!
            //__shared__ bool is_finish[128];
            //is_finish[threadIdx.x] = false;
            //__syncthreads();
            //overall[128*warpId+96+laneId] = small_BU[tid];
            while(__any_sync(0xffffffff, tid < small[0]))
            {
                int count;//t=0;
                bool is_curr ;//= false;
                index_t my_beg; index_t my_end;
                param_t front_vert; 
                //int count=0;  bool is_curr=false;
                bool is_help_curr;

                if(tid>=small[0])
                {
                    //printf("here!!");
                    //overall[128*warpId+96+laneId]=-1;
                    my_finish=true;
                    //is_finish[threadIdx.x]=true;
                    my_beg=-1;my_end=-1;
                }
                else
                {
                    count = 0;
                    is_curr = false;
                    //overall[warpId*128+64+laneId] = 0;
                    //overall[warpId*128+96+laneId] = FQ_BU[tid];
                    //if(tid==2245035) printf("%d %d ", warpId, laneId);  //0 11
                    front_vert = FQ_BU[tid];//overall[128*warpId+96+laneId];//FQ_BU[tid];
                    my_beg = beg_cu[front_vert];
                    my_end = beg_cu[front_vert+1];
                    //if(front_vert==3072563)   printf("here!");//printf("beg_pos=%ld, %ld", my_beg, my_end); 
                }

                //if(tid==2245035) printf("%d %d ", warpId, laneId);  //0 11
                while(__any_sync(0xffffffff, my_beg < my_end))
                    //while(my_beg < my_end)
                {
                    //counts[tid]++;
                    //int index = __popc(__ballot_sync(0xffffffff, my_beg<my_end));
                    if(tid < small[0] && my_beg < my_end)
                        //if(tid < small[0])
                    {
                        //overall[warpId*128+64+laneId]++;
                        param_t neigh = csr_cu[my_beg];
                        param_t status = SA[neigh];
                        //if(FQ_BU[tid]==3072563)  printf("%d ",overall[1*128+96+11]); //1 11
                        //if(overall[128*warpId+96+laneId]==3072563) printf("%d %d ", warpId, laneId);  //2 11
                        //if(front_vert==2304370)  printf("find!%d %d", SA[neigh], neigh); // @@@@
                        if(status==level/*&&my_finish==false*/){
                            SA[front_vert] = level+1;
                            //    if(tid==2245035)  printf("find!"); 
                            count++;                   
                            my_finish = true;
                            //    is_finish[threadIdx.x] = true;                    
                            break;
                        }
                        if(is_curr==false&&count>0)
                            is_curr = (status==level+1);
                    }
                    //if(count==0&&is_curr==true){
                    //    SA[front_vert] = level+2;
                    //    my_finish = true;
                    //    is_finish[threadIdx.x] = true;
                    //}
                    my_beg ++;
                    if(my_beg >= my_end)
                    {
                        //    if(tid==2245035) printf("find!!%ld,%ld,%ld,%ld ", my_beg, my_end, beg_cu[FQ_BU[2245035]], beg_cu[FQ_BU[2245035]+1]); //@@@@
                        my_finish = true;
                        //is_finish[threadIdx.x] = true;
                    }

                    unsigned work = __ballot_sync(0xffffffff, my_finish==true);
                    int finish = __popc(work);
                    //
                    if (finish == 31)
                    {
                        int src_laneId;
                        //shuffle the unfinished one to the rest
                        if(!my_finish)
                        {
                            //    overall[warpId] = laneId;
                            //    overall[wid] = laneId;
                            src_laneId = laneId;
                        }

                        //__syncwarp();
                        //int src_laneId = overall[warpId];
                        //int src_laneId = overall[wid];

                        int my_beg = __shfl_sync(0xffffffff, my_beg, src_laneId) + laneId;
                        int my_end = __shfl_sync(0xffffffff, my_end, src_laneId);

                        //warp-wise computation
                        while(__any_sync(0xffffffff, my_beg < my_end))
                        {
                            param_t SA_neighbor = level-1;
                            if(my_beg < my_end)
                            {
                                param_t my_neighbor = csr_cu[my_beg];
                                SA_neighbor = SA[my_neighbor];
                            }

                            if(__any_sync(0xffffffff, SA_neighbor==level))
                            {
                                if(laneId == 0) SA[front_vert] = level+1;
                                //        if(laneId == src_laneId)
                                //        { 
                                //            SA[front_vert] = level+1;
                                //            count++;
                                //        }
                                break;
                            }

                            if(is_help_curr==false) is_help_curr = (SA_neighbor==level+1);
                            my_beg += 32;
                        }
                        //is_help_curr = __any_sync(0xffffffff, is_help_curr);

                        if(!my_finish)
                        {
                            is_curr = is_help_curr;
                            my_finish = true;
                        }
                        //if((my_beg>=my_end)&&(count==0)){
                        //    if(__any_sync(0xffffffff, is_curr==true)){
                        //        if(laneId==src_laneId)
                        //            SA[front_vert] = level+2;
                        //    }
                        //}
                    }


                    //if(finish>=16&&finish!=32)
                    //{
                    //    A[wid]=0; // finished
                    //    B[wid]=0; // unfinished
                    //    int Id;
                    //    //if(tid==2245035) printf("%d %d %d ", my_finish, SA[FQ_BU[2245035]], FQ_BU[2245035]);//FQ_BU[tid]); //This part has problem!!!

                    //    if(my_finish==true)
                    //    {
                    //        Id = atomicAdd(A + wid, 1);//A[wid];
                    //        //        overall[128*warpId+reverseId] = laneId;//tid;
                    //    }
                    //    else
                    //    {
                    //        Id = B[wid];
                    //        overall[128 * warpId + 32 + atomicAdd(B + wid, 1)] = laneId;//tid;
                    //        //if(tid==2245035)    printf("%d ", Id);
                    //    }

                    //    //printf("here!!");
                    //    //work donation start!!!
                    //    if(Id<32-finish)
                    //    {  
                    //        int lnd1 = overall[128*warpId+32+Id];  //unfinished laneId
                    //        int cnt = overall[128*warpId+64+lnd1];  //assign curr count
                    //        int front = overall[128*warpId+96+lnd1];  //assign unfinished frontier
                    //        //if(overall[128*warpId+96+11]==3072563) printf("here!!warpId=%d ", warpId);  //warpId=0
                    //        //if(tid==2245035)    printf("here!!%d laneId=%d, lnd1=%d,Id=%d ", FQ_BU[tid], laneId, lnd1, Id); //@@@@front=3072563, laneId=11
                    //        //if(front==3072563)  printf("lnd1=%d,Id=%d ", lnd1, Id);
                    //        //if(tid==2245035)    printf("%d lnd1=%d,Id=%d ", front, lnd1, Id);//front=3072540, lnd1=0, Id=7   printf("%d ", overall[128*warpId+96+lnd1]);
                    //        //if(lnd1==11&&Id==7)    printf("%d ", tid);
                    //        int mybeg = (my_finish==true) ? (beg_cu[front]+cnt-1) : (beg_cu[front]+cnt);
                    //        int myend=beg_cu[front+1];
                    //        int cnt1=0;
                    //        bool is_curr1 = false;
                    //        //if(mybeg==myend)
                    //        //    is_finish[32*warpId+lnd1] = true;
                    //        while(mybeg<myend)
                    //        {
                    //            //interleaved working on the remaining neighbors of the unfinished frontier
                    //            cnt++;
                    //            param_t neigh1 = csr_cu[mybeg];
                    //            param_t status1 = SA[neigh1];
                    //            //printf("here!!");
                    //            //if(front==2304370)  printf("find!%d %d ", SA[neigh1], neigh1); 
                    //            //if(status1==level){
                    //            //&&is_finish[32*warpId+lnd1]==false
                    //            if(status1==level)
                    //            {  //threadIdx.x
                    //                //if(front==64842)  printf("find!");
                    //                SA[front] = level+1;
                    //                cnt1++;
                    //                //printf("here!");
                    //                //need communication when terminate
                    //                is_finish[32*warpId+lnd1] = true;
                    //                break;
                    //            }
                    //            if(is_curr1==false)
                    //                is_curr1 = (status1==level+1);
                    //            mybeg += 2;
                    //            if(mybeg>=myend)
                    //                is_finish[32*warpId+lnd1] = true;
                    //        }


                    //        if(cnt1==0&&is_curr1==true)
                    //        {
                    //            //    if(front==2304370)  printf("find!!"); //@@@
                    //            SA[front] = level+2;
                    //            //need communication when terminate
                    //            is_finish[32*warpId+lnd1] = true;
                    //        }
                    //        //if(mybeg>=myend)
                    //        //    is_finish[32*warpId+lnd1] = true;
                    //        }
                    //        //A[wid]=0;B[wid]=0;
                    //    }

                    //    if(tid<small[0])
                    //    {
                    //        my_beg++;
                    //        if(my_beg>=my_end)
                    //        {
                    //            my_finish = true;
                    //            is_finish[threadIdx.x] = true;
                    //        }
                    //        my_beg++;
                    //    }
                }

                if(count==0&&is_curr==true)
                {
                    SA[front_vert] = level+2;
                    //is_finish[threadIdx.x] = true;
                    my_finish = true;
                } 

                //if(__popc(__ballot_sync(0xffffffff, is_finish[threadIdx.x]==true))==32)
                //if(__popc(__ballot_sync(0xffffffff, my_finish==true))==32)
                //{
                if(laneId==0)
                    //wbeg = atomicAdd(counter, 32*(counter[0]<small[0]));
                    wbeg = atomicAdd(counter, 32);
                tid = __shfl_sync(0xffffffff, wbeg, 0, 32) + laneId; 
                //}
                }

            }

            //template<typename param_t, typename index_t>
            //__global__ void small_BU_update(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* counter, param_t padding){
            //    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
            //    param_t tid = TID;
            //    //bool is_stop=false;
            //    while(tid < small[0]/* && is_stop==false*/){
            //        if(tid==TID){
            //            int count=0;
            //            bool is_curr = false;
            //            param_t front_vert = small_BU[tid];
            //            index_t my_beg = beg_cu[front_vert];
            //            index_t my_end = beg_cu[front_vert+1];
            //            int counts=0;
            //            while(my_beg < my_end){
            //                counts++;
            //                param_t neigh = csr_cu[my_beg];
            //                param_t status = SA[neigh];
            //                if(status==level){
            //                    SA[front_vert] = level+1;
            //                    count++;
            //                    break;
            //                }
            //                if(is_curr==false)
            //                    is_curr = (status==level+1);
            //                my_beg++;
            //            }
            //            //-----------comment out to test the benefits of asynchronous traversal---------------
            //            //if(count==0&&is_curr==true)
            //             //   SA[front_vert] = level+2;
            //            
            //             
            //             //tid = atomicAdd(counter, padding*(counter[0]<small[0]));
            //            tid = atomicAdd(counter, padding);
            //        }
            //        else /*if(tid>=blockDim.x*gridDim.x)*/{
            //            for(int i = 0; i < padding; i++){
            //                if(tid < small[0]){
            //                    int count=0;
            //                    bool is_curr = false;
            //                    param_t front_vert = small_BU[tid];
            //                    index_t my_beg = beg_cu[front_vert];
            //                    index_t my_end = beg_cu[front_vert+1];
            //                    while(my_beg < my_end){
            //                        param_t neigh = csr_cu[my_beg];
            //                        param_t status = SA[neigh];
            //                        if(status==level){
            //                            SA[front_vert] = level+1;
            //                            count++;
            //                            break;
            //                        }
            //                        if(is_curr==false)
            //                            is_curr = (status==level+1);
            //                        my_beg++;
            //                    }
            //            //-----------comment out to test the benefits of asynchronous traversal---------------
            //                    //if(count==0&&is_curr==true)
            //                    //    SA[front_vert] = level+2;
            //                    tid++;
            //                }
            //                else{  break;  }
            //            }
            //            //tid = atomicAdd(counter, padding*(counter[0]<small[0]));
            //            tid = atomicAdd(counter, padding);
            //        }
            //        //tid = atomicAdd(counter, counter[0]<small[0]);
            //    }
            //}
            //


            template<typename param_t, typename index_t>
                __global__ void thread_centric_BU_update(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* counter, param_t padding)
                {
                    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t tid = TID;
                    //bool is_stop=false;
                    while(tid < small[0]/* && is_stop==false*/)
                    {
                        for(int i = 0; i < padding; i++)
                        {
                            int count=0;
                            bool is_curr = false;
                            param_t front_vert = small_BU[tid];
                            index_t my_beg = beg_cu[front_vert];
                            index_t my_end = beg_cu[front_vert+1];
                            while(my_beg < my_end)
                            {
                                param_t neigh = csr_cu[my_beg];
                                param_t status = SA[neigh];
                                if(status==level)
                                {
                                    SA[front_vert] = level+1;
                                    count++;
                                    break;
                                }

                                if(is_curr==false)
                                    is_curr = (status==level+1);
                                my_beg++;
                            }
                            //-----------comment out to test the benefits of asynchronous traversal---------------
                            if(count==0&&is_curr==true)
                                SA[front_vert] = level+2;
                            tid++;
                        }

                        tid = atomicAdd(counter, padding);
                    }
                    //tid = atomicAdd(counter, padding*(counter[0]<small[0]));
                }


            template<typename param_t, typename index_t>
                __global__ void warp_centric_BU_update(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, param_t* counter, param_t padding)
                {
                    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t tid = TID;
                    const param_t woff = threadIdx.x & 31;  
                    param_t group_off = 0;

                    //bool is_stop=false;
                    while(tid < small[0]/* && is_stop==false*/)
                    {
                        for(int i = 0; i < padding; i++)
                        {
                            int count=0;
                            bool is_curr = false;
                            param_t front_vert = small_BU[tid];
                            index_t my_beg = beg_cu[front_vert];
                            index_t my_end = beg_cu[front_vert+1];
                            while(my_beg < my_end)
                            {
                                param_t neigh = csr_cu[my_beg];
                                param_t status = SA[neigh];
                                if(status==level)
                                {
                                    SA[front_vert] = level+1;
                                    count++;
                                    break;
                                }

                                if(is_curr==false)
                                    is_curr = (status==level+1);
                                my_beg++;
                            }
                            //-----------comment out to test the benefits of asynchronous traversal---------------
                            if(count==0&&is_curr==true)
                                SA[front_vert] = level+2;
                            tid++;
                        }

                        if(woff == 0) group_off = atomicAdd(counter, (padding<<5));
                        tid = __shfl_sync(0xffffffff, group_off, 0) + woff*padding;
                    }
                    //tid = atomicAdd(counter, padding*(counter[0]<small[0]));
                }




            template<typename param_t, typename index_t>
                __global__ void medium_SA_BU_update(param_t* SA, param_t* medium_BU, param_t* medium, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, index_t* Count, index_t* deg){
                    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t wid = tid >> 5;
                    const param_t woff = tid & 31;
                    while(wid < medium[0])
                    {
                        int count=0;
                        index_t counts = 0;
                        bool is_curr = false;  //indicate whether it has a *current-level neighbor* or not
                        param_t front_vert = medium_BU[wid];
                        index_t my_beg = beg_cu[front_vert] + woff;
                        index_t my_end = beg_cu[front_vert + 1];
                        int gap = (my_end - my_beg);
                        while(__any_sync(0xffffffff, my_beg < my_end))
                        {
                            counts++;
                            param_t SA_neighbor = level-1;
                            if(my_beg < my_end)
                            {
                                param_t my_neighbor = csr_cu[my_beg];
                                SA_neighbor = SA[my_neighbor];
                            }
                            if(__any_sync(0xffffffff, SA_neighbor==level))
                            {
                                if(woff==0){
                                    SA[front_vert] = level+1;
                                    deg[wid] = counts*32;
                                }
                                count++;
                                //if(woff==31)
                                //    deg[wid] = (my_beg - beg_cu[front_vert]);
                                //deg[wid] = counts*32;
                                //counts = ((my_beg - (beg_cu[front_vert] + woff))/32)+1;
                                break;
                            }
                            if(is_curr==false)
                                is_curr = (SA_neighbor==level+1);
                            my_beg += 32;
                        }
                        if((my_beg>=my_end)&&(count==0))
                        {  // make sure the above while loop already finish
                            if(__any_sync(0xffffffff, is_curr==true))
                            {
                                if(woff==0)
                                    SA[front_vert] = level+2;
                            }
                        }
                        //deg[wid] = counts;
                        //Count[TID] += counts;
                        wid += blockDim.x * gridDim.x / 32;
                    }
                    //Count[TID] = my_count;
                    //__syncthreads();
                }

            template<typename param_t, typename index_t>
                __global__ void large_SA_BU_update(param_t* SA, param_t* large_BU, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count, index_t* Count, index_t* deg){
                    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                    param_t bid = blockIdx.x;
                    while(bid < large[0]){
                        int count=0;
                        index_t counts=0;
                        bool is_curr = false;
                        param_t front_vert = large_BU[bid];
                        index_t my_beg = beg_cu[large_BU[bid]] + threadIdx.x;
                        index_t my_end = beg_cu[large_BU[bid]+1];
                        //my_count += (((my_end - my_beg)/blockDim.x)+1);
                        while(__syncthreads_or(my_beg < my_end))
                        {
                            counts++;
                            param_t SA_neighbor = level-1;
                            if(my_beg < my_end)
                            {
                                param_t my_neighbor = csr_cu[my_beg];
                                SA_neighbor = SA[my_neighbor];
                            }
                            if(__syncthreads_or(SA_neighbor==level))
                            {
                                if(threadIdx.x==0)
                                    SA[front_vert] = level+1;
                                count++;
                                deg[bid] = counts*blockDim.x;
                                break;
                            }
                            if(is_curr==false)
                                is_curr = (SA_neighbor==level+1);
                            my_beg += blockDim.x;
                        }
                        if((my_beg>=my_end)&&(count==0))
                        {  // make sure the above while loop already finish
                            if(__syncthreads_or(is_curr==true))
                            {
                                if(threadIdx.x==0)
                                    SA[front_vert] = level+2;
                            }
                        }
                        //Count[TID] += count;
                        bid += gridDim.x;
                    }
                    //Count[TID] = my_count;
                    //__syncthreads();
}

    template<typename param_t, typename index_t>
void BFS_ControlFlow(param_t padding, string dataset, param_t init, param_t adaptive, float* ratio, param_t* tag, double* iter, double& b1, double& b2, double& b3, double& b4, double& b5, double& b6, double& T1, double& T2, double& T3, double& T4, param_t BlockDim, param_t GridDim, param_t* sa, param_t* status, param_t* Status, param_t* SA, param_t* small, param_t* medium, param_t* large, param_t* small_FQ, param_t* medium_FQ, param_t* large_FQ, param_t* small_cnt, param_t* medium_cnt, param_t* large_cnt, param_t* small_CNT, param_t* medium_CNT, param_t* large_CNT, param_t vert_count, param_t* level, param_t* d_level, index_t* beg, index_t* beg_cu, param_t* csr_cu, float alpha, param_t* small_BU, param_t* medium_BU, param_t* large_BU)
{
    int counts = 0;
    for(int i = 0; i < vert_count; i++)
    {
        if(sa[i]!=-1&&Status[i]==-1)
        {
            counts++;
            break;
        }
    }
    int TD_adaptive = INFINITE;
    bool is_change = false;
    index_t* g_deg = new index_t[vert_count];
    for(int i = 0; i < vert_count; i++)
        g_deg[i] = beg[i+1]-beg[i];
    index_t max=workload_gap1(g_deg, vert_count);
    std::cout<<"maximum degree of graph is "<<max<<"\n";
    int hub_num=0;
    int theta = 1000;
    for(int i = 0; i < vert_count; i++)
        if(beg[i+1] - beg[i]>theta)
            hub_num++;
    std::cout<<"hub_num = "<<hub_num<<"\n";
    param_t* base_small = new param_t[1];
    param_t* base_med = new param_t[1];
    param_t* base_large = new param_t[1];
    param_t* small_CNT1; param_t* med_CNT1; param_t* large_CNT1;
    if((beg[init+1]-beg[init])>0 && (beg[init+1]-beg[init])<=32)
    {
        small[0] = init;
        small_cnt[0] = 1;
        base_small[0] = small_cnt[0];
    }
    else if((beg[init+1]-beg[init])>32 && (beg[init+1]-beg[init])<=256)
    {
        medium[0] = init;
        medium_cnt[0] = 1;
        base_med[0] = medium_cnt[0];
    }
    else if((beg[init+1]-beg[init])>256)
    {
        large[0] = init;
        large_cnt[0] = 1;
        base_large[0] = large_cnt[0];
    }
    cudaMalloc((void **)&small_CNT1, sizeof(param_t));
    cudaMalloc((void **)&med_CNT1, sizeof(param_t));
    cudaMalloc((void **)&large_CNT1, sizeof(param_t));
    cudaMemset(small_CNT1, 0, sizeof(param_t));
    cudaMemset(med_CNT1, 0, sizeof(param_t));
    cudaMemset(large_CNT1, 0, sizeof(param_t));
    H_ERR(cudaMemcpy(small_CNT, small_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(medium_CNT, medium_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(large_CNT, large_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(small_FQ, small, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(medium_FQ, medium, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(large_FQ, large, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    param_t* small_FQ1;
    H_ERR(cudaMalloc((void **)&small_FQ1, sizeof(param_t)*(vert_count)));
    cudaMemset(small_FQ1, -1, vert_count * sizeof(param_t));
    param_t* medium_FQ1;
    H_ERR(cudaMalloc((void **)&medium_FQ1, sizeof(param_t)*(vert_count)));
    cudaMemset(medium_FQ1, -1, vert_count * sizeof(param_t));
    param_t* large_FQ1;
    H_ERR(cudaMalloc((void **)&large_FQ1, sizeof(param_t)*(vert_count)));
    cudaMemset(large_FQ1, -1, vert_count * sizeof(param_t));
    int hub_before=0, hub_after=0;
    int res = vert_count % (BlockDim*GridDim);
    param_t* s_count;
    cudaMalloc((void **)&s_count, sizeof(param_t)*(BlockDim*GridDim));
    param_t* m_count;
    cudaMalloc((void **)&m_count, sizeof(param_t)*(BlockDim*GridDim));
    param_t* l_count;
    cudaMalloc((void **)&l_count, sizeof(param_t)*(BlockDim*GridDim));
    param_t* Count = new param_t[BlockDim*GridDim];
    param_t* s_offset;
    cudaMalloc((void **)&s_offset, sizeof(param_t)*(BlockDim*GridDim+1));
    param_t* m_offset;
    cudaMalloc((void **)&m_offset, sizeof(param_t)*(BlockDim*GridDim+1));
    param_t* l_offset;
    cudaMalloc((void **)&l_offset, sizeof(param_t)*(BlockDim*GridDim+1));
    param_t* offset = new param_t[BlockDim*GridDim+1];
    float beta = 0.1;
    index_t* S_count; index_t* M_count; index_t* L_count;
    H_ERR(cudaMalloc((void **)&S_count, sizeof(index_t)*(BlockDim*GridDim)));
    H_ERR(cudaMalloc((void **)&M_count, sizeof(index_t)*(BlockDim*GridDim)));
    H_ERR(cudaMalloc((void **)&L_count, sizeof(index_t)*(BlockDim*GridDim)));
    H_ERR(cudaMemset(S_count, 0, (BlockDim*GridDim) * sizeof(index_t)));
    H_ERR(cudaMemset(M_count, 0, (BlockDim*GridDim) * sizeof(index_t)));
    H_ERR(cudaMemset(L_count, 0, (BlockDim*GridDim) * sizeof(index_t)));
    long int* count = new long int[100];




    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);



    memset(count, 0, 100*sizeof(long int));
    while(counts>0)
    {    //still needs traversal
        //long int count=0; 
        for(int i = 0; i < vert_count; i++)
        {
            if(Status[i]==level[0])  //criteria1: total number of edges of frontiers / total number of edges in the graph
                count[level[0]] += beg[i+1] - beg[i];
        }
        if(is_change==false)
        {
            if(level[0]==0&&count[level[0]]>=INFINITE)   //for src vertex with extreme large degree case, like trackers
            {
                is_change=true;
                TD_adaptive=level[0]+1;
            }
            else if(level[0]>0&&(count[level[0]-1]>=INFINITE&&count[level[0]]>=INFINITE*10&&count[level[0]]/count[level[0]-1]>=8))
            {
                is_change=true;
                TD_adaptive=level[0];
            }
            else if(level[0]>0&&count[level[0]]>=INFINITE*100 || (count[level[0]-1]<INFINITE&&count[level[0]]>=INFINITE*10&&count[level[0]]/count[level[0]-1]>=1000))
            {
                is_change=true;
                TD_adaptive=level[0];
            }
        }
        //std::cout<<"TD_adaptive="<<TD_adaptive<<"\n";
        std::cout<<"\nstarting the "<<level[0]<<"th round....\n"; 
        ratio[level[0]] = (float)count[level[0]]/(float)beg[vert_count];
        std::cout<<"ratio = "<<ratio[level[0]]<<", "<<beg[vert_count]<<"\n";
        if(ratio[level[0]]>=alpha && tag[level[0]-1]>=0)
            hub_before = level[0];
        if(ratio[level[0]]<alpha && tag[level[0]-1]==-1)
            hub_after = level[0];
        //getchar();
        if(ratio[level[0]] < alpha)
        {  //choose top-down 
            std::cout<<"Round "<<level[0]<<" is top-down BFS....\n";
            tag[level[0]] = 0;
            if((hub_before==0&&level[0]<TD_adaptive))  //(hub_after>0&&level[0]>hub_after)
            {
                //if(hub_before==0 || (hub_after>0&&tag[level[0]-1]!=1)){
                std::cout<<"entering scan-free top-down...\n";
                double t1=wtime();
                small_SA_TD_update<param_t, index_t><<<GridDim, BlockDim/*, 0, stream1*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count, small_FQ1, medium_FQ1, large_FQ1, small_CNT1, med_CNT1, large_CNT1);
                medium_SA_TD_update<param_t, index_t><<<GridDim, BlockDim/*, 0, stream2*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count, small_FQ1, medium_FQ1, large_FQ1, small_CNT1, med_CNT1, large_CNT1);
                large_SA_TD_update<param_t, index_t><<<GridDim, BlockDim/*, 0, stream3*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count, small_FQ1, medium_FQ1, large_FQ1, small_CNT1, med_CNT1, large_CNT1);
                H_ERR(cudaDeviceSynchronize());
                double t2 = wtime();
                swap(small_CNT, small_CNT1);
                swap(medium_CNT, med_CNT1);
                swap(large_CNT, large_CNT1);
                double t1_ = wtime();
                cudaMemset(small_CNT1, 0, sizeof(param_t));
                cudaMemset(med_CNT1, 0, sizeof(param_t));
                cudaMemset(large_CNT1, 0, sizeof(param_t));                
                 H_ERR(cudaDeviceSynchronize());//Added anil test for synchronize
           double t2_ = wtime();
                iter[level[0]] = (t2-t1)+(t2_-t1_);
                T2 += (t2 - t1);

                swap(small_FQ, small_FQ1);
                swap(medium_FQ, medium_FQ1);
                swap(large_FQ, large_FQ1);
                double t3=wtime();
                cudaMemset(small_FQ1, -1, vert_count * sizeof(param_t));
                cudaMemset(medium_FQ1, -1, vert_count * sizeof(param_t));
                cudaMemset(large_FQ1, -1, vert_count * sizeof(param_t));
                  H_ERR(cudaDeviceSynchronize());//Added anil test for synchronize
               double t4=wtime();
                T2 += (t4-t3);
            }
            else
            {
                //std::cout<<"entering scan-based top-down...\n";
                if(hub_after>0&&level[0]>hub_after)
                {
                    std::cout<<"entering last phase top-down...\n";
                    double t1 = wtime();
                    small_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim/*, 0, stream1*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
                    medium_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim/*, 0, stream2*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
                    large_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim/*, 0, stream3*/>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
                    H_ERR(cudaDeviceSynchronize());
                    double t2 = wtime();
                    iter[level[0]] = (t2-t1);
                    T2 += (t2 - t1);
                }
                else
                {
                    std::cout<<"entering second phase top-down...\n";
                    tag[level[0]]=1;
                    double t1=wtime();
                    if(tag[level[0]-1]==1)
                    {
                        atomic_FQ_Generator2<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, medium_FQ, large_FQ, small_CNT, medium_CNT, large_CNT, vert_count, level[0], beg_cu);
                        H_ERR(cudaDeviceSynchronize());
                    }
                    double t2=wtime();
                    double t3 = wtime();
                    small_SA_TD_update2<param_t, index_t><<<GridDim, BlockDim/*, 0, stream1*/>>>(SA, small_FQ, small_CNT, beg_cu, csr_cu, level[0], vert_count);
                    medium_SA_TD_update2<param_t, index_t><<<GridDim, BlockDim/*, 0, stream2*/>>>(SA, medium_FQ, medium_CNT, beg_cu, csr_cu, level[0], vert_count);
                    large_SA_TD_update2<param_t, index_t><<<GridDim, BlockDim/*, 0, stream3*/>>>(SA, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
                    H_ERR(cudaDeviceSynchronize());
                    double t4 = wtime();
                    iter[level[0]] = (t4-t3);
                    //if(tag[level[0]-1]==1)
                    T2 += (t4 - t3)+(t2-t1);
                    if(tag[level[0]-1]==0)
                    {
                        std::cout<<"here!!";
                        cudaMemset(small_CNT, 0, sizeof(param_t));
                        cudaMemset(medium_CNT, 0, sizeof(param_t));
                        cudaMemset(large_CNT, 0, sizeof(param_t));
                    }
                }
            }
            H_ERR(cudaMemcpy(Status, SA, sizeof(param_t)*vert_count, cudaMemcpyDeviceToHost));
            int cnt=0;
            std::cout<<"\nlevel"<<level[0]<<" Status array:";
            for(int i = 0; i < vert_count; i++)
            {
                if(Status[i]==level[0])
                {
                    cnt++;
                }
            }
            std::cout<<"\nlevel "<<level[0]<<" contains "<<cnt<<" vertices.\n";
            }
            else
            {   //choose bottom-up
                std::cout<<"\nThis round is doing bottom-up BFS....";
                tag[level[0]] = -1;
                small_cnt[0] = 0; medium_cnt[0] = 0;  large_cnt[0] = 0;
                // double t1=wtime();
                H_ERR(cudaMemcpy(small_CNT, small_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
                H_ERR(cudaMemcpy(medium_CNT, medium_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
                H_ERR(cudaMemcpy(large_CNT, large_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
                   H_ERR(cudaDeviceSynchronize());//Added anil test for synchronize
             double t1=wtime();
                cudaMemset(s_count, 0, (BlockDim*GridDim) * sizeof(param_t));
                cudaMemset(m_count, 0, (BlockDim*GridDim) * sizeof(param_t));
                cudaMemset(l_count, 0, (BlockDim*GridDim) * sizeof(param_t));
                cudaMemset(s_offset, -1, (BlockDim*GridDim+1) * sizeof(param_t));
                cudaMemset(m_offset, -1, (BlockDim*GridDim+1) * sizeof(param_t));
                cudaMemset(l_offset, -1, (BlockDim*GridDim+1) * sizeof(param_t));
                    H_ERR(cudaDeviceSynchronize());//Added anil test for synchronize
             double t2=wtime();
                T3 += (t2-t1);
                double time = 0;
                double t5 = wtime();


                //-
                //--Because we are assigning single thread to all frontiers (small, medium and large), 
                //- there is no need for a classification at frontier queue generation.
                //--


                //BU_FQ_find<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, medium_FQ, large_FQ, small_CNT, medium_CNT,  large_CNT, beg_cu, level[0], vert_count);
                ballot_counting<param_t, index_t><<<GridDim, BlockDim>>>(SA, vert_count, res, s_count, m_count, l_count, beg_cu);
                H_ERR(cudaDeviceSynchronize());
                double t6 = wtime();
                my_prefixsum<param_t>(Count, s_count, BlockDim, GridDim, offset, s_offset, time);
                double t9 = wtime();
                FQ_gen<param_t, index_t><<<GridDim, BlockDim>>>(SA, vert_count, res, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, s_offset, m_offset, l_offset, s_count, m_count, l_count, beg_cu);
                H_ERR(cudaDeviceSynchronize());
                double t10 = wtime();
                b1 += (t6 - t5);   b2 += time; b3 += (t10 - t9);
                T3 += ((t6 - t5) + time + (t10 - t9));
                H_ERR(cudaMemcpy(small_cnt, small_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
                H_ERR(cudaMemcpy(medium_cnt, medium_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
                H_ERR(cudaMemcpy(large_cnt, large_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));

                index_t* Deg1; //index_t* Deg2; index_t* Deg3; 
                H_ERR(cudaMalloc((void **)&Deg1, sizeof(index_t)*(small_cnt[0])));
                H_ERR(cudaMemset(Deg1, -1, small_cnt[0] * sizeof(index_t)));
                param_t* d_counter;
                param_t* counter = new param_t[1];
                counter[0] = padding*BlockDim*GridDim;
                //counter[0] = BlockDim*GridDim;
                H_ERR(cudaMalloc((void **)&d_counter, sizeof(param_t)));
                H_ERR(cudaMemcpy(d_counter, counter, sizeof(param_t), cudaMemcpyHostToDevice));
                 H_ERR(cudaDeviceSynchronize());//Added anil test for synchronize

                double t7 = wtime();

                //thread-centric
                thread_centric_BU_update<param_t, index_t><<<GridDim, BlockDim, 0, stream1>>>(SA, small_FQ, small_CNT, beg_cu, csr_cu, level[0], vert_count, d_counter, padding);
                H_ERR(cudaDeviceSynchronize());
                double t8 = wtime();
                T4 += (t8 - t7);
                iter[level[0]] = (t8-t7)+(t6-t5)+time+(t10-t9); 
               if(tag[level[0]-1]==1)
                {
                }
                H_ERR(cudaMemcpy(Status, SA, sizeof(param_t)*vert_count, cudaMemcpyDeviceToHost));
            }       
            level[0]++;
            counts = 0;
            int mismatch=0;
            int real_num=0;
            for(int i = 0; i < vert_count; i++)
            {
                if(sa[i]==level[0]&&Status[i]!=level[0])
                {
                    mismatch++;
                    //        std::cout<<Status[i]<<" ";  //e.g.3072563
                }
                if(sa[i]!=-1&&Status[i]==-1)
                {
                    counts++;
                }
                if(sa[i]==level[0])
                    real_num++;
            }
            if(mismatch>0)
                std::cout<<"\nNext level "<<level[0]<<" mismatch: \n"<<"MISMATCH !!!!!!!!!!!!!!!!!!!!"<<mismatch<<"\n"; 
            std::cout<<"real_num = "<<real_num;
        }
    }

