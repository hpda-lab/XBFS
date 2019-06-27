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
//#include "workload_gap.cuh"

void workload_distribution(long int* arr, int n, long int max, long int min){
    int* count = new int[max-min+1];
    int* cc = new int[max - min+1];
    //std::cout<<"range = "<<max-min+1<<"\n";
    for(int i = 0; i < max-min+1; i++){
        count[i] = 0;
        cc[i] = 0;
    }
    for(int i = 0; i < n; i++){
        int val = arr[i]-min;
        count[val]++;
    }
    int** index = new int*[max-min+1];
    for(int i = 0; i < max-min+1; i++)
        index[i] = new int[count[i]];
    for(int i = 0; i < n; i++){
        int val = arr[i] - min;
        index[val][cc[val]++] = i;
    }
    std::cout<<"\nThread workload distribution: \n";
    for(int i = 0; i < max-min+1; i++){
        std::cout<<i+min<<": ";
        for(int j = 0; j < count[i]; j++){
            std::cout<<index[i][j]<<" ";
        }
        std::cout<<"\n";
    }
    int total=0;
    for(int i = 0; i < max-min+1; i++){
    //    std::cout<<count[i]<<" ";
        total += count[i];
    }
    std::cout<<"total = "<<total<<"\n";
}
