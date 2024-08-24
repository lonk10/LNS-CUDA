#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"
#include "../include/util.cuh"

#define THREADS_PER_BLOCK 256

// Given k partitions and n*m/100 threads per block
// each threads check if the destr_mask[threadIdx.x] node is present in its block's 
// partition and destroys it if necessary
// usage should be destroy<<k, n*m/100>>
// costs update should be handled by another function

__global__ void destroy(int *parts, int *destr_mask, int m){
    int tid = threadIdx.x;
    if (tid < m){
        int node = destr_mask[threadIdx.x];
        int ind = blockIdx.x * blockDim.x + node;
        if (parts[ind] == 1){
            parts[ind] = 0;
        }
    }
}

// Assigns n*m/100 nodes to random partions

__global__ void assignToParts(int n, int node, int *parts, float *result, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    //should be a parallel reduction here
    int k = blockIdx.x;
    int ind = threadIdx.x;
    extern __shared__ int sdata_i[];
    extern __shared__ int sdata_e[];

    int start_r = csr_rep -> offsets[node];
    int end_r = csr_rep -> offsets[node+1];
    int start_c = csc_rep -> offsets[node];
    int end_c = csc_rep -> offsets[node+1];

    int r_size = end_r - start_r;
    int c_size = end_c - start_c;
    int max_size = r_size > c_size ? r_size : c_size;
    if (ind == 0){
        cudaMalloc( (void**)&sdata_i, max_size * sizeof(int));
        cudaMalloc( (void**)&sdata_e, max_size * sizeof(int));
    }
    __syncthreads(); // wait for allocation
    // initialization
    sdata_i[ind] = 0;
    sdata_e[ind] = 0;
    __syncthreads();

    // gather values
    int edge_node;
    if (ind < r_size){
        edge_node = csr_rep -> col_indexes[start_r + ind];
        if (parts[k*n+edge_node]){
            sdata_i[ind] = csr_rep -> values[start_r + ind];
        } else {
            sdata_e[ind] = csr_rep -> values[start_r + ind];
        }
    }
    if (ind < r_size){
        edge_node = csc_rep -> row_indexes[start_c + ind];
        if (parts[k*n+edge_node]){
            sdata_i[ind] = csc_rep -> values[start_c + ind];
        } else {
            sdata_e[ind] = csc_rep -> values[start_c + ind];
        }
    }
    __syncthreads();

    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (ind < stride){
            sdata_i[ind] += sdata_i[ind + stride];
            sdata_e[ind] += sdata_e[ind + stride];
        }
        __syncthreads();
    }

    // store final result
    if (ind == 0){
        int mu_k = 2*(int_costs[k] + sdata_i[0]);
        result[k] = 100*(mu_k / (mu_k + ext_costs[k] + sdata_e[0]));
    }
}

__host__ void repair(int *parts, int k, int *destr_mask, int n, int m, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    //int i = 0;
    int node;
    float *result;
    cudaMalloc( (void**)&result, k * sizeof(float));
    int asgn;
    int temp_cost;
    for (int i = 0; i < (n*m/100); i++){
        //k = asgn_mask[i];
        node = destr_mask[i];
        assignToParts<<<k, THREADS_PER_BLOCK>>>(n, node, parts, result, int_costs, ext_costs, csr_rep, csc_rep);
        cudaDeviceSynchronize();
        asgn = 0;
        temp_cost = result[0];
        for (int j = 0; j < k; j++){
            if (result[j] > temp_cost){
                asgn = j;
                temp_cost = result[j];
            }
        }
        
        //actual assign
        parts[asgn*n+node] == 1;
    }
    cudaFree(result);
}

__global__ void resetMask(int *mask, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size){
        mask[tid] = 0;
    }
}

void lns_v1(int *in_parts, int *weights, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR *row_rep, CSC *col_rep){
    int *best = (int *) malloc(nodes_num*parts_num*sizeof(int));
    for (int i = 0; i < nodes_num*parts_num; i++){
        best[i] = in_parts[i];
    }
    //compute node costs
    int *temp_int_cost, *temp_ext_cost;

    int *int_cost = (int *)malloc(parts_num*sizeof(int));
    int *ext_cost = (int *)malloc(parts_num*sizeof(int));
    cudaMalloc( (void**)&temp_int_cost, parts_num * sizeof(int));
    cudaMalloc( (void**)&temp_ext_cost, parts_num * sizeof(int));
    computeAllEdgeCost(best, row_rep, col_rep, parts_num, nodes_num, edges_num, int_cost, ext_cost);
    for (int i = 0; i < parts_num; i++){
        printf("init node cost %d \ninit edge cost %d\n", int_cost[i], ext_cost[i]);
    }
    float best_cost = computeCost(int_cost, ext_cost, parts_num);
    float new_cost;
    int destr_nodes = nodes_num*m/100;
    int *d_destr_mask, *temp;
    int *destr_mask = (int *) malloc(destr_nodes * sizeof(int));
    cudaMalloc( (void**)&d_destr_mask, destr_nodes * sizeof(int));
    cudaMalloc( (void**)&temp, nodes_num * parts_num * sizeof(int));
    srand(time(NULL));

    printf("Initial cost is: %f\n", best_cost);

    for (int iter = 0; iter < MAX_ITER; iter++){
        //printf("Iteration %d start\n", iter);
        //reset values
        for (int i = 0; i < destr_nodes; i++){
            destr_mask[i] = 0;
        }
        //resetMask<<<parts_num, THREADS_PER_BLOCK>>>(destr_mask, destr_nodes);
        cudaMemcpy(temp, in_parts, nodes_num*parts_num*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_int_cost, int_cost, parts_num*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_ext_cost, ext_cost, parts_num*sizeof(int), cudaMemcpyHostToDevice);

        //printf("Destroy step %d\n", iter);
        //destroy step
        computeRandomMask(destr_mask, nodes_num, m);
        cudaMemcpy(d_destr_mask, destr_mask, destr_nodes*sizeof(int), cudaMemcpyHostToDevice);
        destroy<<<parts_num, THREADS_PER_BLOCK>>>(temp, destr_mask, m);
        cudaDeviceSynchronize();

        //printf("Repair step %d\n", iter);
        //repair step
        //computeRandomAssignment(asgn_mask, nodes_num, m, parts_num);
        repair(temp, parts_num, destr_mask, nodes_num, m, temp_int_cost, temp_ext_cost, row_rep, col_rep);

        //printf("Accept step %d\n", iter);
        //accept step
        if (checkMass(temp, weights, parts_num, nodes_num, max_mass)){
            new_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num);
            if (new_cost > best_cost)
            printf("New best cost is: %f\n", new_cost);
                best_cost = new_cost;
                memcpy(best, temp, nodes_num*parts_num*sizeof(int));
        }
    }
    printf("Final cost is: %f\n", best_cost);
    printf("Partitions were:\n");
    for (int i = 0; i < parts_num; i++){
        printf("Partition %d : ", i);
        for (int j = 0; j < nodes_num; j++){
            printf("%d", in_parts[i*nodes_num+j]);
        }
        printf("\n");
    }
    printf("Partitions are now:\n");
    for (int i = 0; i < parts_num; i++){
        printf("Partition %d : ", i);
        for (int j = 0; j < nodes_num; j++){
            printf("%d", best[i*nodes_num+j]);
        }
        printf("\n");
    }
    //printf("snip:\n");
    cudaFree(destr_mask);
    //printf("snapp:\n");
    //free(asgn_mask);
    //printf("snoop:\n");
}