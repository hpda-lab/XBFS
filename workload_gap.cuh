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

void workload_gap(long int* arr, int n){
    long int max = arr[0];
    long int min = arr[0];
    long int total = 0;
    for(int i = 1; i < n; i++){
        if(arr[i]>max)
            max = arr[i];
        if(arr[i]<min)
            min = arr[i];
        total += arr[i];
    }
    std::cout<<"max workload is:"<<max<<", min workload is:"<<min<<", and gap is: "<<max-min<<"\n";
    std::cout<<"total workload is:"<<total;
}
long int workload_gap1(long int* arr, int n){
    long int max = arr[0];
    int index = arr[0];
    for(int i = 1; i < n; i++){
        if(arr[i]>max){
            max = arr[i];
            index = i;
        }
    }
    std::cout<<"\nvertexID = "<<index<<" ";
    return max;
}
long int workload_gap2(long int* arr, int n){
    long int max = arr[0];
    long int min = arr[0];
    for(int i = 1; i < n; i++){
        if(arr[i]<min)
            min = arr[i];
    }
    return min;
}
