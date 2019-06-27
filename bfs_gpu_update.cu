#include <iostream>
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <list>
#include <vector>
#include "wtime.h"
#include <assert.h>
#include <fstream>
#include <math.h>

#include "bfs_adaptiveFQ_wb_async.cuh"

using namespace std;
int max_level(long int* arr, int n){
    long int val = arr[0];
    int idx = 0;
    for(int i = 1; i < n; i++){
        if(arr[i]>val){
            val = arr[i];
            idx = i;
        }
    }
    return idx;
}
int main(int args, char **argv)
{
	//typedef char data_t;//typedef int data_t;
    typedef int param_t;
    typedef long int index_t;
	std::cout<<"Input: ./exe beg csr weight BlockDim GridDim gpu_id results alpha src workload padding\n";
	if(args!=12){std::cout<<"Wrong input\n"; return -1;}
	
	const char *beg_file=argv[1];
	const char *csr_file=argv[2];
	const char *weight_file=argv[3];
    	
	param_t BlockDim = atoi(argv[4]);//int BlockDim = atoi(argv[1]);
    param_t GridDim = atoi(argv[5]);//int GridDim = atoi(argv[2]);
    //index_t total = atoi(argv[6]);//int iterNum = atoi(argv[3]);
    param_t id = atoi(argv[6]);
    const char *filename = argv[7];
    float alpha = (float)atof(argv[8]);
    param_t src = atoi(argv[9]);
    string dataset = argv[10];
    param_t padding = atoi(argv[11]);
    std::cout<<"alpha = "<<alpha<<"\n";
	//template <file_vertex_t, file_index_t, file_weight_t
	//new_vertex_t, new_index_t, new_weight_t>
	graph<int, long, int, int, long, char>
	*ginst = new graph
	<int, long, int, int, long, char>
	(beg_file,csr_file,weight_file);
   
    ofstream file(filename, ios::app);
    ofstream file1("Hybrid_iteration.csv", ios::app);
    //ofstream file(filename);
/*    ofstream edge("edge-list");
    for(int i = 0; i < ginst->vert_count; i++){
        edge<<i<<"'s neighor list: ";
        for(long int j = ginst->beg_pos[i]; j < ginst->beg_pos[i+1]; j++)
            edge<<ginst->csr[j]<<" ";
        edge<<"\n";
    }
*/    
    //std::cout<<"64842's degree: "<<ginst->beg_pos[64843]-ginst->beg_pos[64842]<<"\n";
    //std::cout<<"64842's beg_pos: "<<ginst->beg_pos[64842]<<" end_pos: "<<ginst->beg_pos[64843]<<"\n";
    //for(int i = ginst->beg_pos[64842]; i < ginst->beg_pos[64842+1]; i++)
    //    if(ginst->csr[i]==957364)
    //        std::cout<<"find!!!!!\n";
    //    std::cout<<ginst->csr[i]<<" ";
    //You can implement your single threaded graph algorithm here.
    //like BFS, SSSP, PageRank and etc.
    int* degree = new int[3];
    memset(degree, 0, 3 * sizeof(int));
    std::cout<<"Vertex classification based on the degree: \n";
    for(int i = 0; i < ginst->vert_count; i++){
        if(ginst->beg_pos[i+1] - ginst->beg_pos[i]<=32)
            degree[0]++;
        else if((ginst->beg_pos[i+1] - ginst->beg_pos[i])>32 && (ginst->beg_pos[i+1] - ginst->beg_pos[i])<=256){
            degree[1]++;
        }
        else if(ginst->beg_pos[i+1] - ginst->beg_pos[i]>256){
            degree[2]++;
        }
    }
    std::cout<<"small degree counts: "<<degree[0]<<" "<<"medium degree counts: "<<degree[1]<<" "<<"large degree counts: "<<degree[2];
    param_t init = 0;
    //for(int i = 0; i < ginst->vert_count; i++)
    //{
    //    if(ginst->beg_pos[i]>0){
    //       init = i - 1;
    //       break;
    //    }
    //}
    init = src;
    //std::cout<<"first 10 vertices' degree: \n";
    //for(int i = 0; i < 10; i++){
    //    std::cout<<ginst->beg_pos[i+1] - ginst->beg_pos[i]<<" ";
    //}
    std::cout<<"\nStarting BFS graph traversal from node "<<init<<"......\n";    
    int k = 0;
    param_t* status = new param_t[ginst->vert_count];
    for(int i = 0; i < ginst->vert_count; i++){
        status[i] = -1;
    }
    std::cout<<"The status array: ";
    //for(int i = 0; i < ginst->vert_count; i++)
    //    std::cout<<status[i]<<" ";
    bool isdone = false;
    status[init] = 0;
   //BFS traverse from level k to level k+1
    for(k = 0; isdone != true; k++){
        isdone = true;
        for(int i = 0; i < ginst->vert_count; i++){
            if(status[i]==k){
               for(long int j = ginst->beg_pos[i]; j < ginst->beg_pos[i+1]; j++){
                   if(status[ginst->csr[j]]==-1){
                      isdone = false;
                      status[ginst->csr[j]] = k+1;
                   }
               }
            }
        }
    }  
    std::cout<<"\nk = "<<k<<"\n";
    int invalid = 0;
    for(int i = 0; i < ginst->vert_count; i++){
        if(status[i]==-1)
            invalid++;
    //    if(status[i]==k-1)
    //    std::cout<<status[i]<<" ";
    }
    std::cout<<"\n#of remaining -1: "<<invalid<<"\n";
    std::cout<<"\nstarting processing the status array on GPU....\n";
    cudaSetDevice(id);
    double* iter = new double[k];
    memset(iter, 0, (k)*sizeof(double));
    float* ratio = new float[k];
    memset(ratio, 0, (k)*sizeof(float));
    //param_t* front = new param_t[k];
    //memset(front, 0, (k)*sizeof(param_t));
    //for(int i = 0; i < k; i++){
    //    for(int j = 0; j < ginst->vert_count; j++){
    //        if(status[j]==i){
    //            front[status[j]]++;
    //        }
    //    }
    //}
    //std::cout<<"number of frontiers in each level: ";
    //for(int i = 0; i < k; i++)
    //    std::cout<<front[i]<<" ";
    //index_t* edge = new index_t[k];
    //memset(edge, 0, (k)*sizeof(index_t));
    //for(int i = 0; i < ginst->vert_count; i++){
    //    for(int j = 0; j < k; j++){
    //        if(status[i]==j){
    //            edge[j] += ginst->beg_pos[i+1] - ginst->beg_pos[i];
    //        }
    //    }
    //}
    //std::cout<<"\n number of edges in each level: ";
    //for(int i = 0; i < k; i++)
    //    std::cout<<edge[i]<<" ";
    //int temp2 = max_level(edge, k) - 1;
    //int adaptive = temp2;
    //std::cout<<"\nadaptive = "<<adaptive<<"\n";
    //for(int i = 0; i < temp2; i++)
    //    std::cout<<"adaptive-1/="<<(float)edge[i+1]/(float)edge[i]<<"   ";
    int adaptive=0;
    std::cout<<"The average degree of the graph is: "<<ginst->beg_pos[ginst->vert_count]/ginst->vert_count;
//================================================= 1. Atomic-based version start... ===========================================
    index_t* beg_cu;
    H_ERR(cudaMalloc((void **)&beg_cu, sizeof(index_t)*(ginst->vert_count+1)));
    H_ERR(cudaMemcpy(beg_cu, ginst->beg_pos, sizeof(index_t)*(ginst->vert_count+1), cudaMemcpyHostToDevice));
    param_t* csr_cu;
    H_ERR(cudaMalloc((void **)&csr_cu, sizeof(param_t)*(ginst->beg_pos[ginst->vert_count])));
    H_ERR(cudaMemcpy(csr_cu, ginst->csr, sizeof(param_t)*(ginst->beg_pos[ginst->vert_count]), cudaMemcpyHostToDevice));
    
    param_t* status_cpp;
    H_ERR(cudaMalloc((void **)&status_cpp, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(status_cpp, status, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));

    param_t* small = new param_t[ginst->vert_count];
    memset(small, -1, ginst->vert_count * sizeof(param_t));
    param_t* medium = new param_t[ginst->vert_count];
    memset(medium, -1, ginst->vert_count * sizeof(param_t));
    param_t* large = new param_t[ginst->vert_count];
    memset(large, -1, ginst->vert_count * sizeof(param_t));
    
    param_t* small_FQ;
    H_ERR(cudaMalloc((void **)&small_FQ, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(small_FQ, small, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));
    param_t* medium_FQ;
    H_ERR(cudaMalloc((void **)&medium_FQ, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(medium_FQ, medium, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));
    param_t* large_FQ;
    H_ERR(cudaMalloc((void **)&large_FQ, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(large_FQ, large, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));

    param_t* small_BU;
    H_ERR(cudaMalloc((void **)&small_BU, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(small_BU, small, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));
    param_t* medium_BU;
    H_ERR(cudaMalloc((void **)&medium_BU, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(medium_BU, medium, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));
    param_t* large_BU;
    H_ERR(cudaMalloc((void **)&large_BU, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(large_BU, large, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));

    param_t* Status = new param_t[ginst->vert_count];
    memset(Status, -1, ginst->vert_count * sizeof(param_t));
    Status[init] = 0;
    param_t* Status_cu;
    H_ERR(cudaMalloc((void **)&Status_cu, sizeof(param_t)*(ginst->vert_count)));
    H_ERR(cudaMemcpy(Status_cu, Status, sizeof(param_t)*(ginst->vert_count), cudaMemcpyHostToDevice));
    
    param_t* small_cnt = (param_t*)malloc(sizeof(param_t));
    *small_cnt = 0;
    param_t* small_CNT;
    H_ERR(cudaMalloc((void**)&small_CNT, sizeof(param_t)));
    H_ERR(cudaMemcpy(small_CNT, small_cnt, sizeof(param_t), cudaMemcpyHostToDevice));    
    param_t* medium_cnt = (param_t*)malloc(sizeof(param_t));
    *medium_cnt = 0;
    param_t* medium_CNT;
    H_ERR(cudaMalloc((void**)&medium_CNT, sizeof(param_t)));
    H_ERR(cudaMemcpy(medium_CNT, medium_cnt, sizeof(param_t), cudaMemcpyHostToDevice));
    param_t* large_cnt = (param_t*)malloc(sizeof(param_t));
    *large_cnt = 0;
    param_t* large_CNT;
    H_ERR(cudaMalloc((void**)&large_CNT, sizeof(param_t)));
    H_ERR(cudaMemcpy(large_CNT, large_cnt, sizeof(param_t), cudaMemcpyHostToDevice));

    param_t* level = (param_t*)malloc(sizeof(param_t));
    *level = 0;
    param_t* d_level;
    H_ERR(cudaMalloc((void**)&d_level, sizeof(param_t)*1));
    H_ERR(cudaMemcpy(d_level, level, sizeof(param_t)*1, cudaMemcpyHostToDevice));
    //getchar(); 
    param_t* tag = new param_t[k];
    memset(tag, -1, k*sizeof(param_t));
    double T1 = 0, T2 = 0, T3 = 0, T4 = 0;
    double t1 = 0, t2 = 0, t3 = 0;
    double t4 = 0, t5 = 0, t6 = 0;
    double* T = new double[3];
    //for(int i = 0; i < 3; i++)
    //    T[i]=0;
    double naivetime=wtime();
    BFS_ControlFlow<param_t, index_t>(padding, dataset, init, adaptive, ratio, tag, iter, t1, t2, t3, t4, t5, t6, T1, T2, T3, T4, BlockDim, GridDim, status, status_cpp, Status, Status_cu, small, medium, large, small_FQ, medium_FQ, large_FQ, small_cnt, medium_cnt, large_cnt, small_CNT, medium_CNT, large_CNT, ginst->vert_count, level, d_level, ginst->beg_pos, beg_cu, csr_cu, alpha, small_BU, medium_BU, large_BU);
    cudaDeviceSynchronize();
    cout<<endl<<"naivetime="<<(wtime()-naivetime)*1000 <<" ms"<<endl;
    double end = wtime(); 
    H_ERR(cudaMemcpy(Status, Status_cu, sizeof(param_t)*ginst->vert_count, cudaMemcpyDeviceToHost));
    std::cout<<"\nThe GPU-based status[]: \n";
    int mismatch=0;
    //assert(memcmp(Status, status, sizeof(param_t)*ginst->vert_count)==0);
    //std::cout<<"We finished<<<<<<<<<<<<<<<<<<\n"; 
    
    for(int i = 0; i < ginst->vert_count; i++) {
        if(Status[i]!=status[i]){
            mismatch++;
      //      std::cout<<Status[i]<<" ";     
    //        std::cout<<i<<" ";
        }
    }
    std::cout<<"\n\nCurrent level = "<<level[0]<<" "<<mismatch<<" padding="<<padding;
    std::cout<<"\nTime consumption of top-down && bottom up BFS: "<<T1+T2+T3+T4<<" seconds from the starting vertex "<<init<<"\n";
    file1<<dataset<<": ";
    for(int i=0; i<k; i++)
        file1<<tag[i]<<" ";
    file<<"\n";
    std::cout<<"\ntop-down FQ_gen: "<<T1<<" seconds. with counting: "<<t4<<", prefix-offset: "<<t5<<", FQ_gen: "<<t6<<" seconds.\n";
    std::cout<<"top-down SA update: "<<T2<<" seconds.\n";
    std::cout<<"bottom-up FQ_gen: "<<T3<<" seconds. with ballot_counting: "<<t1<<", prefix-offset: "<<t2<<", FQ_gen: "<<t3<<" seconds.\n";
    std::cout<<"bottom-up SA update: "<<T4<<" seconds.\n";
    std::cout<<"top-down: "<<T1+T2<<" seconds.\n";
    std::cout<<"bottom-up: "<<T3+T4<<" seconds.\n";
  
      //anil 1.20.2019
     ofstream Timefile("time_Baseline+adaptive+FQ+wb+async_P6000_Camera_more_synch.csv", ios::app);
      cout<<dataset<<";"<<"Total time of traversal (ms)="<<T1+T2+T3+T4<<endl;
Timefile<<dataset<<";"<<T1+T2<<";"<<T3+T4<<";"<<T1+T2+T3+T4<<endl;
     Timefile.close();
    //~1.20.2019

//std::cout<<"hub_level time consumption:"<<T[0]<<", "<<T[1]<<", "<<T[2]<<"\n";
    //file<<"This is scan-free version...\n";
    //file<<"Iteration time_consumption #frontiers with alpha="<<alpha<<" and total time: "<<T1+T2+T3+T4<<" and bottom-up FQ_gen: "<<T3<<"\n";
    //file<<init<<","<<ginst->vert_count-invalid<<","<<alpha<<","<<T1+T2<<","<<T1<<","<<T2<<","<<T1*100/(T1+T2)<<"%"<<endl<<endl;
    /*for(int i = 0; i < k; i++){
        if(tag[i]==0)
            file<<"TD_ratio:"<<ratio[i]<<" "<<iter[i]<<"\n";
        else if(tag[i]==1)
            file<<"BU_ratio:"<<ratio[i]<<" "<<iter[i]<<"\n";
    }*/
    //file<<filename<<","<<T1+T2+T3+T4<<","<<padding<<"\n";
    
    file.close();
    cudaFree(Status_cu); 
    cudaFree(small_FQ);
    cudaFree(medium_FQ);
    cudaFree(large_FQ);
    std::cout<<"Now we finish!\n";
 //   getchar();
    return 0;
}

