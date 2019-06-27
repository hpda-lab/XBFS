#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

//static void HandleError( cudaError_t err, const char *file, int line ) {
//       if (err != cudaSuccess) {
//           printf( "\n%s in %s at line %d\n", \
//           cudaGetErrorString( err ), file, line );
//           exit( EXIT_FAILURE );
//        }
//      }
//#define H_ERR( err ) \
//      (HandleError( err, __FILE__, __LINE__ ))

using namespace std;

template<typename param_t, typename index_t>
__global__ void small_SA_TD_update1(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    param_t count= (small[0] < vert_count) ? small[0] : vert_count;
    while(tid < count){  
        param_t front_vert = small_FQ[tid];
        if(SA[front_vert]==level){
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
                        small_FQ[atomicAdd(small, 1)] = front;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ[atomicAdd(medium, 1)] = front;
                    }
                    else if((next_end-next_beg)>256){
                        large_FQ[atomicAdd(large, 1)] = front;
                    }
                }
            }
            my_beg++;
        }}
        tid += blockDim.x * gridDim.x;
    }
//    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void medium_SA_TD_update1(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    param_t wid = tid >> 5;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    param_t count= (medium[0] < vert_count) ? medium[0] : vert_count;
    const param_t woff = tid & 31;
    while(wid < count){
        param_t front_vert = medium_FQ[wid];
        if(SA[front_vert]==level){
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
                        small_FQ[atomicAdd(small, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ[atomicAdd(medium, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>256){
                         large_FQ[atomicAdd(large, 1)] = vertex;
                    }
                }
            }
            my_beg += 32; 
        }}
        wid += blockDim.x * gridDim.x / 32;
    }
//    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void large_SA_TD_update1(param_t* SA, param_t* small_FQ, param_t* small, param_t* medium_FQ, param_t* medium, param_t* large_FQ, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){ 
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;    
    param_t bid = tid / blockDim.x;
    param_t m1=small[0], m2=medium[0], m3=large[0];
    const param_t count= (large[0] < vert_count) ? large[0] : vert_count;
    while(bid < count){
        param_t front_vert = large_FQ[bid];
        if(SA[front_vert]==level){
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
                        small_FQ[atomicAdd(small, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>32 && (next_end-next_beg)<=256){
                        medium_FQ[atomicAdd(medium, 1)] = vertex;
                    }
                    else if((next_end-next_beg)>256){
                        large_FQ[atomicAdd(large, 1)] = vertex;
                    }
                }
            }
            my_beg += blockDim.x;
        }}   
        bid += gridDim.x;
    }
//    __syncthreads();
}

template<typename param_t, typename index_t>
__global__ void BU_FQ_find1(param_t* SA, param_t* small_BU, param_t* medium_BU, param_t* large_BU, param_t* small, param_t* medium, param_t* large, index_t* beg_cu, param_t level, param_t vert_count){
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
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void small_SA_BU_update1(param_t* SA, param_t* small_BU, param_t* small, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
    param_t tid = TID;
    while(tid < small[0]){    
        int count=0;
        param_t front_vert = small_BU[tid];
        index_t my_beg = beg_cu[front_vert];
        index_t my_end = beg_cu[front_vert+1];
        while(my_beg < my_end){
            count++;
            param_t status = SA[csr_cu[my_beg]];
            if(status==level){
                SA[front_vert] = level+1;
                break;
            }
            my_beg++;
        }
        //Count[TID] += count; 
        tid += blockDim.x*gridDim.x;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void medium_SA_BU_update1(param_t* SA, param_t* medium_BU, param_t* medium, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    const param_t TID = threadIdx.x + blockIdx.x * blockDim.x;
    param_t tid = TID;
    param_t wid = tid >> 5;
    const param_t woff = tid & 31;
    while(wid < medium[0]){ 
        int count=0;
        param_t front_vert = medium_BU[wid];
        index_t my_beg = beg_cu[front_vert] + woff;
        index_t my_end = beg_cu[front_vert + 1];
        while(__any_sync(0xffffffff, my_beg < my_end)){
            count++;
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
        //Count[TID] += count;
        wid += blockDim.x * gridDim.x / 32;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
__global__ void large_SA_BU_update1(param_t* SA, param_t* large_BU, param_t* large, index_t* beg_cu, param_t* csr_cu, param_t level, param_t vert_count){
    param_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    //param_t bid = tid / blockDim.x;
    param_t bid = blockIdx.x;
    while(bid < large[0]){
        param_t front_vert = large_BU[bid];
        index_t my_beg = beg_cu[large_BU[bid]] + threadIdx.x;
        index_t my_end = beg_cu[large_BU[bid]+1];
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
        bid += gridDim.x;
    }
    //__syncthreads();
}

template<typename param_t, typename index_t>
void BFS_ControlFlow1(param_t padding, string dataset, param_t init, param_t adaptive, float* ratio, param_t* tag, double* iter, double& b1, double& b2, double& b3, double& b4, double& b5, double& b6, double& T1, double& T2, double& T3, double& T4, param_t BlockDim, param_t GridDim, param_t* sa, param_t* status, param_t* Status, param_t* SA, param_t* small, param_t* medium, param_t* large, param_t* small_FQ, param_t* medium_FQ, param_t* large_FQ, param_t* small_cnt, param_t* medium_cnt, param_t* large_cnt, param_t* small_CNT, param_t* medium_CNT, param_t* large_CNT, param_t vert_count, param_t* level, param_t* d_level, index_t* beg, index_t* beg_cu, param_t* csr_cu, float alpha, param_t* small_BU, param_t* medium_BU, param_t* large_BU){
    int counts = 0;
    for(int i = 0; i < vert_count; i++){
        if(sa[i]!=-1&&Status[i]==-1){
            counts++;
            break;
        }
    }
    if((beg[init+1]-beg[init])>0 && (beg[init+1]-beg[init])<=32){
        small[0] = init;
        small_cnt[0] = 1;
    }
    else if((beg[init+1]-beg[init])>32 && (beg[init+1]-beg[init])<=256){
        medium[0] = init;
        medium_cnt[0] = 1;
    }
    else if((beg[init+1]-beg[init])>256){
        large[0] = init;
        large_cnt[0] = 1;
    }
    H_ERR(cudaMemcpy(small_CNT, small_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(medium_CNT, medium_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(large_CNT, large_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(small_FQ, small, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(medium_FQ, medium, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(large_FQ, large, sizeof(param_t)*(vert_count), cudaMemcpyHostToDevice));
    while(counts>0){    //still needs traversal
        int count = 0;
        for(int i = 0; i < vert_count; i++){
            if(Status[i]==level[0])
                count += beg[i+1] - beg[i];
        }
        //getchar();
        //std::cout<<"\nstarting the "<<level[0]<<"th round....\n"; 
        //std::cout<<"\ncurrent threshold indicator: "<<(float)count/(float)beg[vert_count]<<"\n";
        if((float)count/(float)beg[vert_count] < alpha){  //choose top-down 
            //cudaMemset(small_CNT, 0, sizeof(param_t));
            //cudaMemset(medium_CNT, 0, sizeof(param_t));
            //cudaMemset(large_CNT, 0, sizeof(param_t));
            //std::cout<<"Round "<<level[0]<<" is top-down BFS....\n";
            tag[level[0]] = 0;
            //getchar(); 
            //H_ERR(cudaMemcpy(small, small_FQ, sizeof(param_t)*(vert_count), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(medium, medium_FQ, sizeof(param_t)*(vert_count), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(large, large_FQ, sizeof(param_t)*(vert_count), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(small_cnt, small_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(medium_cnt, medium_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(large_cnt, large_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //std::cout<<"\n\nCurrent level "<<level[0]<<" small queue: ";
            //for(int i = 0; i < *small_cnt; i++){
            //    if(small[i]!=-1)
            //    std::cout<<small[i]<<" ";
            //}
            //std::cout<<"\n\nCurrent level "<<level[0]<<" medium queue: ";
            //for(int i = 0; i < *medium_cnt; i++){
            //    if(medium[i]!=-1)
            //    std::cout<<medium[i]<<" ";
            //}
            //std::cout<<"\n\nCurrent level "<<level[0]<<" large queue: ";
            //for(int i = 0; i < *large_cnt; i++){
            //    if(large[i]!=-1)
            //    std::cout<<large[i]<<" ";
            //}
            //cudaStream_t stream1, stream2, stream3;
            //cudaStreamCreate(&stream1);
            //cudaStreamCreate(&stream2);
            //cudaStreamCreate(&stream3);
            double t1 = wtime();
            small_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
            medium_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
            large_SA_TD_update1<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, small_CNT, medium_FQ, medium_CNT, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
            //H_ERR(cudaStreamSynchronize(stream1));
            //H_ERR(cudaStreamSynchronize(stream2));
            //H_ERR(cudaStreamSynchronize(stream3));
            H_ERR(cudaDeviceSynchronize());
            double t2 = wtime();
            iter[level[0]] = (t2-t1);
            T2 += (t2 - t1);
            //qsort(small, vert_count, sizeof(param_t), cmp);
            //qsort(medium, vert_count, sizeof(param_t), cmp);
            //qsort(large, vert_count, sizeof(param_t), cmp); 
            H_ERR(cudaMemcpy(Status, SA, sizeof(param_t)*vert_count, cudaMemcpyDeviceToHost));
            //int cnt=0;
            //std::cout<<"\n\nlevel"<<level[0]<<" Status array:\n";
            //for(int i = 0; i < vert_count; i++){
            //    if(Status[i]==level[0])
            //        std::cout<<i<<" ";
            //    cnt++;
            //}
            //std::cout<<"level "<<level[0]<<" contains "<<cnt<<" vertices.\n";
        }
	    else{   //choose bottom-up
            //std::cout<<"\nThis round is doing bottom-up BFS....";
            tag[level[0]] = 1;
            //getchar();
            //std::cout<<"\nBefore BU_FQ_find, current status array: \n";
            //H_ERR(cudaMemcpy(Status, SA, sizeof(param_t)*vert_count, cudaMemcpyDeviceToHost)); 
            small_cnt[0] = 0; medium_cnt[0] = 0;  large_cnt[0] = 0;
            H_ERR(cudaMemcpy(small_CNT, small_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(medium_CNT, medium_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(large_CNT, large_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
            double t5 = wtime();
            //BU_FQ_find<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_BU, medium_BU, large_BU, small_CNT, medium_CNT,  large_CNT, beg_cu, level[0], vert_count);
            BU_FQ_find1<param_t, index_t><<<GridDim, BlockDim>>>(SA, small_FQ, medium_FQ, large_FQ, small_CNT, medium_CNT,  large_CNT, beg_cu, level[0], vert_count);
            H_ERR(cudaDeviceSynchronize());
            double t6 = wtime();
            T3 += (t6 - t5);
            //std::cout<<"\n\nAfter BU_FQ_find,  "<<level[0]<<" small queue: ";
            //for(int i = 0; i < vert_count; i++){
            //    if(small[i]!=-1)
            //        std::cout<<small[i]<<" ";
            //}
            //std::cout<<"\n\nAfter BU_FQ_find,  "<<level[0]<<" medium queue: ";
            //for(int i = 0; i < vert_count; i++){
            //    if(medium[i]!=-1)
            //        std::cout<<medium[i]<<" ";
            //}
            //std::cout<<"\n\nAfter BU_FQ_find,  "<<level[0]<<" large queue: ";
            //for(int i = 0; i < vert_count; i++){
            //    if(large[i]!=-1)
            //        std::cout<<large[i]<<" ";
            //}
            //H_ERR(cudaMemcpy(small_cnt, small_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(medium_cnt, medium_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //H_ERR(cudaMemcpy(large_cnt, large_CNT, sizeof(param_t), cudaMemcpyDeviceToHost));
            //std::cout<<"\nSmall frontier count: "<<small_cnt[0]<<", Medium frontier count: "<<medium_cnt[0]<<", Large frontier count: "<<large_cnt[0];
            //getchar();
            double t7=wtime();
            cudaStream_t stream1, stream2, stream3;
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            cudaStreamCreate(&stream3);
            //double t7 = wtime();
            //small_SA_BU_update<param_t, index_t><<<GridDim, BlockDim, 0, stream1>>>(SA, small_BU, small_CNT, beg_cu, csr_cu, level[0], vert_count); 
            small_SA_BU_update1<param_t, index_t><<<GridDim, BlockDim, 0, stream1>>>(SA, small_FQ, small_CNT, beg_cu, csr_cu, level[0], vert_count);
            //std::cout<<"After small_SA_BU_update...\n";
            //medium_SA_BU_update<param_t, index_t><<<GridDim, BlockDim, 0, stream2>>>(SA, medium_BU, medium_CNT, beg_cu, csr_cu, level[0], vert_count);
            medium_SA_BU_update1<param_t, index_t><<<GridDim, BlockDim, 0, stream2>>>(SA, medium_FQ, medium_CNT, beg_cu, csr_cu, level[0], vert_count);
            //std::cout<<"After medium_SA_BU_update...\n";
            //large_SA_BU_update<param_t, index_t><<<GridDim, BlockDim, 0, stream3>>>(SA, large_BU, large_CNT, beg_cu, csr_cu, level[0], vert_count);  
            large_SA_BU_update1<param_t, index_t><<<GridDim, BlockDim, 0, stream3>>>(SA, large_FQ, large_CNT, beg_cu, csr_cu, level[0], vert_count);
            //std::cout<<"After large_SA_BU_update...\n";
            H_ERR(cudaStreamSynchronize(stream1));
            H_ERR(cudaStreamSynchronize(stream2));
            H_ERR(cudaStreamSynchronize(stream3));
            //H_ERR(cudaDeviceSynchronize());
            double t8 = wtime();
            iter[level[0]] = (t8-t7)+(t6-t5);
            T4 += (t8 - t7);
            H_ERR(cudaMemcpy(Status, SA, sizeof(param_t)*vert_count, cudaMemcpyDeviceToHost));
            //std::cout<<"\nlevel"<<level[0]<<" Status array:\n";
            //int cnt=0;
            //for(int i = 0; i < vert_count; i++){
            //    if(Status[i]==level[0])
            //        std::cout<<i<<" ";
            //    cnt++;
            //} 
            //std::cout<<"level "<<level[0]<<" contains "<<cnt<<" vertices.\n";
        }       
        level[0]++;
        counts = 0;
        //std::cout<<"\nNext level "<<level[0]<<" Status array: \n";
        for(int i = 0; i < vert_count; i++){
            //if(sa[i]==level[0]&&sa[i]!=Status[i])
        //    std::cout<<Status[i]<<" ";
            if(sa[i]!=-1&&Status[i]==-1){
            //if(sa[i]!=-1&&sa[i]!=Status[i]){
                counts++;
        //      break;
            }
        }
        //std::cout<<"\n";
    }
}

