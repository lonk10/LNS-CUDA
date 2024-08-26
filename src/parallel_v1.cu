#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"
#include "../include/util.cuh"

#define THREADS_PER_BLOCK 256
#define GRIDS 10
#define BLOCKS_PER_GRID 256

__global__ void removeFromCosts(int *parts, int node, int partition, int n, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep, int *block_sums_i, int *block_sums_e){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int sdata[];
    int start_r = csr_rep -> offsets[node];
    int end_r = csr_rep -> offsets[node+1];
    int start_c = csc_rep -> offsets[node];
    int end_c = csc_rep -> offsets[node+1];

    int r_size = end_r - start_r;
    int c_size = end_c - start_c;
    int max_size = r_size > c_size ? r_size : c_size;

    if (ind < max_size){

        if (tid == 0){
            block_sums_i[blockIdx.x] = 0;
            block_sums_e[blockIdx.x] = 0;
        }
        sdata[tid] = 0;
        __syncthreads();
        int edge_node;
        if (ind < r_size){
            edge_node = csr_rep -> col_indexes[start_r + ind];
            if (parts[partition*n+edge_node] == 1){
                sdata[ind] = csr_rep -> values[start_r + ind];
            } else {
                sdata[ind+blockDim.x] = csr_rep -> values[start_r + ind];
            }
            
        }
        if (ind < c_size){
            edge_node = csc_rep -> row_indexes[start_c + ind];
            if (parts[partition*n+edge_node] == 1){
                sdata[ind] = csc_rep -> values[start_c + ind];
            } else {
                sdata[ind+blockDim.x] = csc_rep -> values[start_c + ind];
            }
        }
        // reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if (ind < stride && (ind + stride) < max_size){
                sdata[ind] += sdata[ind + stride];
                sdata[ind+blockDim.x] += sdata[ind + blockDim.x + stride];
            }
            __syncthreads();
        } /*
        if (ind == 0){
            printf("Hello, thread %d of block %d reduction done\n", ind, k);
        }*/
    
        if (tid == 0){
            atomicAdd(&block_sums_i[blockIdx.x], sdata[0]);
            atomicAdd(&block_sums_e[blockIdx.x], sdata[blockDim.x]);
        }
        if (ind == 0){
            int final_sum_i = 0;
            int final_sum_e = 0;
            for (int i = 0; i < gridDim.x; i++) {
                final_sum_i += block_sums_i[i];
                final_sum_e += block_sums_e[i];
            }
            int_costs[partition] -= final_sum_i;
            ext_costs[partition] -= final_sum_e;
        }
    }
}

// Given k partitions and n*m/100 threads per block
// each threads check if the destr_mask[threadIdx.x] node is present in its block's 
// partition and destroys it if necessary
// usage should be destroy<<k, n*m/100>>
// costs update should be handled by another function

__global__ void destroy(int *parts, int *destr_mask, int destr_nodes, int k, int n, int *int_costs, int *ext_costs, CSR *row_rep, CSC *col_rep){
    int tid = threadIdx.x;
    if (tid < destr_nodes){
        int node = destr_mask[tid];
        int ind = blockIdx.x * blockDim.x + node;
        int *block_sums_i, *block_sums_e;
        cudaMalloc( (void**)&block_sums_i, 256 * sizeof(int));
        cudaMalloc( (void**)&block_sums_e, 256 * sizeof(int));
        if (parts[ind] == 1){
            parts[ind] = 0;
            //remove this
            //use a destr_node sized mask to register where the node was
            //or get it for in_parts (k checks needed), could stay serial tbh
            //them call removeFromCosts for every node with the part as parameter
            removeFromCosts<<<256, 256, 512>>>(parts, node, blockIdx.x, n, int_costs, ext_costs, row_rep, col_rep, block_sums_i, block_sums_e);
            cudaDeviceSynchronize();
            printf("Thread %d of block %d destroyed node %d in partition %d\n", tid, blockIdx.x, node, blockIdx.x);
        }
        cudaFree(block_sums_i);
        cudaFree(block_sums_e);
    }
}
// Assigns n*m/100 nodes to random partions

__global__ void assignToParts(int n, int node, int *parts, float *result, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep, int *block_sums_i, int *block_sums_e){
    //should be a parallel reduction here
    int k = blockIdx.y;
    int block_id = blockIdx.x;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];

    int start_r = csr_rep -> offsets[node];
    int end_r = csr_rep -> offsets[node+1];
    int start_c = csc_rep -> offsets[node];
    int end_c = csc_rep -> offsets[node+1];

    int r_size = end_r - start_r;
    int c_size = end_c - start_c;
    int max_size = r_size > c_size ? r_size : c_size;

    // init of block sums
    // block_sums[k*gridDim.x...k*gridDim+block_id] is the reduction result of block block_id in row k
    if (ind == 0 && block_id == 0){
        block_sums_i[k * gridDim.x + block_id] = 0;
        block_sums_e[k * gridDim.x + block_id] = 0;
    }
    __syncthreads();

    // sdata init
    if (ind < max_size) {
        sdata[ind] = 0;
        sdata[ind + blockDim.x] = 0;
    }

    // gather values
    int edge_node;
    /*
    if (ind == 0 && block_id == 0){
        printf("Hello, thread %d of block %d in row %d, trying to assign node %d\n", ind, block_id, blockIdx.y, node);
        for (int i = 0; i < n; i++){
            printf("Partition[%d][%d] is %d\n", k, i, parts[k*n+i]);
        }
        __syncthreads();
    }*/

    __syncthreads();
    if (ind < r_size){
        edge_node = csr_rep -> col_indexes[start_r + ind];
        if (parts[k*n+edge_node] == 1){
            sdata[ind] = csr_rep -> values[start_r + ind];
            //printf("Edge node %d is in part %d, sdata_i[%d] set to %d\n", edge_node, k, ind, sdata_i[ind]);
        } else {
            sdata[ind+blockDim.x] = csr_rep -> values[start_r + ind];
            //printf("Edge node %d is not in part %d, sdata_e[%d] set to %d\n", edge_node, k, ind, sdata_e[ind]);
        }
        printf("Outgoing edge node %d, part %d, sdata_i[%d] = %d, sdata_e[%d] = %d\n", edge_node, k, ind, sdata[ind], ind, sdata[ind+blockDim.x]);
        
    }
    if (ind < c_size){
        edge_node = csc_rep -> row_indexes[start_c + ind];
        if (parts[k*n+edge_node] == 1){
            sdata[ind] = csc_rep -> values[start_c + ind];
        } else {
            sdata[ind+blockDim.x] = csc_rep -> values[start_c + ind];
        }
        printf("Incoming edge node %d, part %d, sdata_i[%d] = %d, sdata_e[%d] = %d\n", edge_node, k, ind, sdata[ind], ind, sdata[ind+blockDim.x]);
    }
    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (ind < stride && (ind + stride) < max_size){
            sdata[ind] += sdata[ind + stride];
            sdata[ind+blockDim.x] += sdata[ind + blockDim.x + stride];
        }
        __syncthreads();
    } /*
    if (ind == 0){
        printf("Hello, thread %d of block %d reduction done\n", ind, k);
    }*/

    if (ind == 0){
        atomicAdd(&block_sums_i[k * gridDim.x + block_id], sdata[0]);
        atomicAdd(&block_sums_e[k * gridDim.x + block_id], sdata[blockDim.x]);
    }

    // store final result
    if (ind == 0 && block_id == 0){
        int final_sum_i = 0;
        int final_sum_e = 0;
        for (int i = 0; i < gridDim.x; i++) {
            final_sum_i += block_sums_i[k * gridDim.x + i];
            final_sum_e += block_sums_e[k * gridDim.x + i];
        }
        int mu_k = 2*(int_costs[k] + final_sum_i);
        result[k] = 100*((float) mu_k / (float)(mu_k + ext_costs[k] + final_sum_e));
        printf("mu_k: %d idata: %d edata: %d result: %f \n", mu_k, final_sum_i, final_sum_e, result[k]);
    }
}

// assigns node to partition with maximum score
__global__ void assignToBestPart(int k, float *results, int n, int node, int *parts){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];
    if (tid < k){ // reduction for finding index of max value in results
        sdata[tid] = tid; //initialize sdata to partition ids
        __syncthreads();
        int nextTid;
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            nextTid = sdata[tid + stride];
            if (tid < stride){
                if (results[sdata[tid]] < results[nextTid])
                    sdata[tid] = nextTid;
            }
            __syncthreads();
        }
        if (tid == 0){
            parts[sdata[0]*n+node] = 1; // assign node to index sdata[0]
            printf("Assigned node %d to part %d\n", node, sdata[0]);
        }
    }

}

void repair(int *parts, int k, int *destr_mask, int n, int edges_num, int m, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    //int i = 0;
    int node;
    float *d_result;
    cudaMalloc( (void**)&d_result, k * sizeof(float));
    float *result = (float *) malloc(k*sizeof(float));
    int asgn;
    float temp_cost;
    dim3 gridDim(256, k, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    int *block_sums_e, *block_sums_i;
    cudaMalloc( (void**)&block_sums_e, 5 * 256 * sizeof(int));
    cudaMalloc( (void**)&block_sums_i, 5 * 256 * sizeof(int));
    for (int i = 0; i < (n*m/100); i++){
        //k = asgn_mask[i];
        node = destr_mask[i];
        assignToParts<<<gridDim, blockDim, 2 * THREADS_PER_BLOCK * sizeof(int)>>>(n, node, parts, d_result, int_costs, ext_costs, csr_rep, csc_rep, block_sums_i, block_sums_e);
        cudaDeviceSynchronize();
        //debug stuff
        cudaMemcpy(result, d_result, k*sizeof(float), cudaMemcpyDeviceToHost);
        for (int z = 0; z < k; z++){
            printf("result[%d]: %f\n", z, result[z]);
        }
        cudaDeviceSynchronize();
        assignToBestPart<<<1, k, k*sizeof(int)>>>(k, d_result, n, node, parts);
        cudaDeviceSynchronize();
    }
    free(result);
    cudaFree(d_result);
    cudaFree(block_sums_e);
    cudaFree(block_sums_i);
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
    int *d_temp_int_cost, *d_temp_ext_cost;

    int *int_cost = (int *)malloc(parts_num*sizeof(int));
    int *ext_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_int_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_ext_cost = (int *)malloc(parts_num*sizeof(int));
    cudaMalloc( (void**)&d_temp_int_cost, parts_num * sizeof(int));
    cudaMalloc( (void**)&d_temp_ext_cost, parts_num * sizeof(int));
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


    // copy CSR / CSC to device
    CSR *d_row_rep;
    CSC *d_col_rep;
    int *row_offsets, *col_offsets, *col_indexes, *row_indexes, *row_values, *col_values;
    printf("Allocation d_row, d_col\n");
    cudaMalloc( (void**)&d_row_rep, sizeof(CSR));
    cudaMalloc( (void**)&row_offsets, (nodes_num + 1) * sizeof(int));
    cudaMalloc( (void**)&col_indexes, edges_num * sizeof(int));
    cudaMalloc( (void**)&row_values, edges_num * sizeof(int));
    cudaMalloc( (void**)&d_col_rep, sizeof(CSC));
    cudaMalloc( (void**)&col_offsets, (nodes_num + 1) * sizeof(int));
    cudaMalloc( (void**)&row_indexes, edges_num * sizeof(int));
    cudaMalloc( (void**)&col_values, edges_num * sizeof(int));
    printf("Copying temps\n");
    cudaMemcpy(&(d_row_rep->offsets), &row_offsets, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_row_rep->col_indexes), &col_indexes, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_row_rep->values), &row_values, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_col_rep->offsets), &col_offsets, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_col_rep->row_indexes), &row_indexes, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_col_rep->values), &col_values, sizeof(int*), cudaMemcpyHostToDevice);
    printf("Copying into temps\n");
    cudaMemcpy(row_offsets, row_rep->offsets, (nodes_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indexes, row_rep->col_indexes, edges_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_values, row_rep->values, edges_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_offsets, col_rep->offsets, (nodes_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_indexes, col_rep->row_indexes, edges_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_values, col_rep->values, edges_num * sizeof(int), cudaMemcpyHostToDevice);
    



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
        cudaMemcpy(d_temp_int_cost, int_cost, parts_num*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_ext_cost, ext_cost, parts_num*sizeof(int), cudaMemcpyHostToDevice);

        //printf("Destroy step %d\n", iter);
        //destroy step
        computeRandomMask(destr_mask, nodes_num, m);
        cudaMemcpy(d_destr_mask, destr_mask, destr_nodes*sizeof(int), cudaMemcpyHostToDevice);
        destroy<<<parts_num, nodes_num>>>(temp, d_destr_mask, destr_nodes, parts_num, nodes_num, d_temp_int_cost, d_temp_ext_cost, d_row_rep, d_col_rep);
        cudaDeviceSynchronize();
        //removeCosts<<<>>>(d_destr_mask, nodes_num, d_row_rep, d_col_rep)
        cudaDeviceSynchronize();

        //printf("Repair step %d\n", iter);
        //repair step
        //computeRandomAssignment(asgn_mask, nodes_num, m, parts_num);
        repair(temp, parts_num, destr_mask, nodes_num, edges_num, m, d_temp_int_cost, d_temp_ext_cost, d_row_rep, d_col_rep);

        //printf("Accept step %d\n", iter);
        //accept step
        cudaMemcpy(temp_int_cost, d_temp_int_cost, parts_num*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_ext_cost, d_temp_ext_cost, parts_num*sizeof(int), cudaMemcpyDeviceToHost);
        if (checkMass(temp_int_cost, parts_num, max_mass)){
            new_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num);
            printf("New cost found is: %f\n", new_cost);
            if (new_cost > best_cost)
            printf("New best cost is: %f\n", new_cost);
                best_cost = new_cost;
                cudaMemcpy(best, temp, nodes_num*parts_num*sizeof(int), cudaMemcpyDeviceToHost);
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
    
    //free
    free(best);
    free(int_cost);
    free(ext_cost);
    free(temp_int_cost);
    free(temp_ext_cost);
    free(destr_mask);
    //cudafree
    cudaFree(d_temp_int_cost);
    cudaFree(d_temp_ext_cost);
    cudaFree(d_destr_mask);
    cudaFree(temp);
    cudaFree(row_offsets);
    cudaFree(col_indexes);
    cudaFree(row_values);
    cudaFree(col_offsets);
    cudaFree(row_indexes);
    cudaFree(col_values);
    cudaFree(d_row_rep->offsets);
    cudaFree(d_row_rep->col_indexes);
    cudaFree(d_row_rep->values);
    cudaFree(d_col_rep->offsets);
    cudaFree(d_col_rep->row_indexes);
    cudaFree(d_col_rep->values);
    cudaFree(d_row_rep);
    cudaFree(d_col_rep);
}