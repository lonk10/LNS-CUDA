#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"
#include "../include/util.cuh"
#include "../include/parallel_v1.cuh"

#define THREADS_PER_BLOCK 1024
#define GRIDS 10
#define BLOCKS_PER_ROW 1024

__device__ static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        //exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ void warpReduce_v0(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ void warpReduce(volatile int* sdata, int tid, int blockdim) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
    sdata[tid + blockdim] += sdata[tid + blockdim + 32];
    sdata[tid + blockdim] += sdata[tid + blockdim + 16];
    sdata[tid + blockdim] += sdata[tid + blockdim + 8];
    sdata[tid + blockdim] += sdata[tid + blockdim + 4];
    sdata[tid + blockdim] += sdata[tid + blockdim + 2];
    sdata[tid + blockdim] += sdata[tid + blockdim + 1];
}

__inline__ __device__ int warpReduceSum_v0(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// removes the edge cost of nodes in destr_mask
// from their corresponding partition in destr_parts
__global__ void removeNodes(int* parts, int* destr_mask, int n, int* int_costs, int* ext_costs, CSR* csr_rep, CSC* csc_rep, int* block_sums_i, int* block_sums_e, int* removed_nodes) {
    // indexes
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];

    // variables
    int node = destr_mask[blockIdx.y];
    int partition = parts[node];

    // nodes in outgoing edges
    int start_r = csr_rep->offsets[node];
    // nodes in incoming edges
    int start_c = csc_rep->offsets[node];

    int r_size = csr_rep->offsets[node + 1] - start_r;
    int c_size = csc_rep->offsets[node + 1] - start_c;

    if (threadIdx.x == 0) {
        block_sums_i[blockIdx.y * gridDim.x + blockIdx.x] = 0;
        block_sums_e[blockIdx.y * gridDim.x + blockIdx.x] = 0;
    }
    if (ind == 0) {
        removed_nodes[node] = 1;
    }
    __syncthreads();

    // go ahead only if there are enough nodes to handle
    if (ind < r_size + c_size) {
        // initialize block sums
        sdata[threadIdx.x] = 0;
        sdata[threadIdx.x + blockDim.x] = 0;
        __syncthreads();

        // gather edge weights
        int edge_node;
        if (ind < r_size) {
            edge_node = csr_rep->col_indexes[start_r + ind];
            sdata[threadIdx.x + (parts[edge_node] != partition) * blockDim.x] = csr_rep->values[start_r + ind] + csr_rep->values[start_r + ind] * (parts[edge_node] == partition) * (!removed_nodes[edge_node]);

        }
        else if (ind < r_size + c_size) {
            edge_node = csc_rep->row_indexes[start_c + ind - r_size];
            sdata[threadIdx.x + (parts[edge_node] != partition) * blockDim.x] = csc_rep->values[start_c + ind - r_size] + csc_rep->values[start_c + ind - r_size] * (parts[edge_node] == partition) * (!removed_nodes[edge_node]);
        }
        // reduction

        for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
            if (threadIdx.x < stride && (threadIdx.x + stride) < r_size + c_size) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            warpReduce_v0(sdata, threadIdx.x);
            warpReduce_v0(sdata, threadIdx.x + blockDim.x);
        }

        // store block reduction result
        // this is needed because a single can have max 1024 threads
        // so if a node is connected to more than 1024 other nodes, multiple blocks are needed to handle it
        // as shared memory is only intra-block, global memory is needed

        if (threadIdx.x == 0) {
            atomicSub(&int_costs[partition], sdata[0]);
            atomicSub(&ext_costs[partition], sdata[blockDim.x]);
        }
    }
}

__global__ void removeNodes2(int* parts, int* nodes, int n, int* int_costs, int* ext_costs, 
                             CSR *csr_rep, CSC *csc_rep,
                             int* block_sums_i, int* block_sums_e, int* removed_nodes) {
    
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];

    
    int node = nodes[blockIdx.y];
    int start_r = csr_rep -> offsets[node];
    int start_c = csc_rep -> offsets[node];
    int r_size = csr_rep -> offsets[node+1] - start_r;
    int c_size = csc_rep -> offsets[node+1] - start_c;
    int k = parts[node];
    if (ind == 0) {
        removed_nodes[node] = 1;
    }
    // init of block sums
    // block_sums[k*gridDim.x...k*gridDim+block_id] is the reduction result of block block_id in row k
    // sdata init
    sdata[threadIdx.x] = 0;
    sdata[threadIdx.x+blockDim.x] = 0;

    // gather values
    //int s_ind;
    __syncthreads();

    /*if (ind < r_size) {
        edge_node = r_indexes[start_r + ind];
        //printf("node %d edge_node %d s_ind %d\n", node, edge_node, s_ind);
        sdata[2*threadIdx.x + (parts[edge_node] != k)] = r_values[start_r + ind] * (1 + (parts[edge_node] == k) * (!removed_nodes[edge_node]));

    }*/
    int edge_node;
    if (ind < r_size){
        edge_node = csr_rep -> col_indexes[start_r + ind];
        if (parts[edge_node] == k){
            if (removed_nodes[edge_node] == 1){ // this check is needed when multiple nodes from the same partition are removed, as to avoid removing the same edge twice
                sdata[ind] = (csr_rep -> values[start_r + ind]);
            } else {
                sdata[ind] = 2*(csr_rep -> values[start_r + ind]);
            }
        } else {
            sdata[ind+blockDim.x] = csr_rep -> values[start_r + ind];
        }
    }
    if (ind < c_size){
        edge_node = csc_rep -> row_indexes[start_c + ind];
        if (parts[edge_node] == k){
            sdata[ind] = csc_rep -> values[start_c + ind];
            if (removed_nodes[edge_node] == 1){
                sdata[ind] += (csc_rep -> values[start_c + ind]);
            } else {
                sdata[ind] += 2*(csc_rep -> values[start_c + ind]);
            }
        } else {
            sdata[ind+blockDim.x] = csc_rep -> values[start_c + ind];
        }
    }
    /*
    else if (ind < r_size + c_size) {
        edge_node = c_indexes[start_c + ind - r_size];
        //printf("node %d edge_node %d s_ind %d\n", node, edge_node, s_ind);
        sdata[2*threadIdx.x + (parts[edge_node] != k)] = c_values[start_c + ind - r_size] * (1 + (parts[edge_node] == k) * (!removed_nodes[edge_node]));
    }*/
    //printf("gathered %d and %d\n", sdata[threadIdx.x], sdata[threadIdx.x+blockDim.x]);
    __syncthreads();
    // reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < r_size + c_size) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            sdata[threadIdx.x+blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    } 
    __syncthreads();
    if (threadIdx.x < 32) {
        warpReduce_v0(sdata, threadIdx.x);
        warpReduce_v0(sdata, threadIdx.x+blockDim.x);
    }

    if (threadIdx.x == 0) {
        //printf("gathered %d and %d\n", sdata[0], sdata[blockDim.x]);
        block_sums_i[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
        block_sums_e[blockIdx.y * gridDim.x + blockIdx.x] = sdata[blockDim.x];
    }
}

__device__ float computePartCost_v0(float u, float ext) {
    return 100 * (u / (u + ext));
}

// uses 54 registers per thread, 27648 per block out of 65536
// should use around 130+ registers per thread for increase in performance
// 14.75% synch, 12.82% memory dep, 57.70% execution dep
// 10% branch divergence
__global__ void assignToParts(int n, int* nodes, int* parts, int* int_costs, int* ext_costs, CSR* csr_rep, CSC* csc_rep, int* block_sums_i, int* block_sums_e, float *result) {
    //should be a parallel reduction here
    int k = blockIdx.y;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];

    int node = nodes[blockIdx.z];
    int start_r = csr_rep->offsets[node];
    int start_c = csc_rep->offsets[node];

    int r_size = csr_rep->offsets[node + 1] - start_r;
    int c_size = csc_rep->offsets[node + 1] - start_c;

    // init of block sums
    // block_sums[k*gridDim.x...k*gridDim+block_id] is the reduction result of block block_id in row k
    if (threadIdx.x == 0) {
        block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + blockIdx.x] = 0;
        block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + blockIdx.x] = 0;
    }
    // sdata init
    if (threadIdx.x < r_size + c_size) {
        sdata[threadIdx.x] = 0;
        sdata[threadIdx.x + blockDim.x] = 0;
    }

    // gather values
    int edge_node;
    int s_ind;
    __syncthreads();
    if (ind < r_size) {
        edge_node = csr_rep->col_indexes[start_r + ind];
        s_ind = (parts[edge_node] != k) * blockDim.x;
        sdata[threadIdx.x + s_ind] = csr_rep->values[start_r + ind];

    }
    else if (ind < r_size + c_size) {
        edge_node = csc_rep->row_indexes[start_c + ind - r_size];
        s_ind = (parts[edge_node] != k) * blockDim.x; //used to know if value is be stored as ext or int
        sdata[threadIdx.x + s_ind] = csc_rep->values[start_c + ind - r_size];
    }
    // reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < r_size + c_size) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    } if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x, blockDim.x);

    if (ind == 0) {
        atomicAdd(&block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + blockIdx.x], sdata[0]);
        atomicAdd(&block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + blockIdx.x], sdata[blockDim.x]);
    }

    if (ind == 0 && blockIdx.x == 0) {
        int final_sum_i = 0;
        int final_sum_e = 0;
        for (int i = 0; i < gridDim.x; i++) { // fare reduction
            final_sum_i += block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + i];
            final_sum_e += block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + i];
        }
        int mu_k = (int_costs[k] + final_sum_i);
        float old_cost = computePartCost_v0(int_costs[k], ext_costs[k]);
        float new_cost = computePartCost_v0(mu_k, ext_costs[k] + final_sum_e);
        result[blockIdx.z * gridDim.y + k] = new_cost - old_cost;
    }
}

// removes nodes in destr_mask from 
// removes nodes in destr_mask from 
void destroy(int* parts, int* destr_mask, int destr_nodes, int k, int n, int* int_costs, int* ext_costs, CSR* row_rep, CSC* col_rep) {
    int* block_sums_i, * block_sums_e, * destr_parts, * removed_nodes;
    cudaMalloc((void**)&block_sums_i, destr_nodes * BLOCKS_PER_ROW * sizeof(int));
    cudaMalloc((void**)&block_sums_e, destr_nodes * BLOCKS_PER_ROW * sizeof(int));
    cudaMalloc((void**)&destr_parts, destr_nodes * sizeof(int));
    dim3 dest_grid(destr_nodes, k, 1);
    // get partitions of destroyed nodes
    //getPartitionPerDestrNode << <dest_grid, THREADS_PER_BLOCK >> > (parts, destr_mask, destr_parts, destr_nodes, n);
    dim3 grid_dim(BLOCKS_PER_ROW, destr_nodes, 1);
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    cudaMalloc((void**)&removed_nodes, n * sizeof(int));
    // set to zero removed_nodes
    setToZero << <n / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (removed_nodes, n);
    cudaDeviceSynchronize();
    // remove nodes
    removeNodes2 << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (parts, destr_mask, n, int_costs, ext_costs, row_rep, col_rep, block_sums_i, block_sums_e, removed_nodes);
    cudaDeviceSynchronize(); // probably not needed
    updatePartWeights << <destr_nodes, BLOCKS_PER_ROW, 2 * BLOCKS_PER_ROW * sizeof(int)>> > (destr_mask, parts, block_sums_i, block_sums_e, int_costs, ext_costs);
    cudaDeviceSynchronize();
    cudaFree(block_sums_i);
    cudaFree(block_sums_e);
    cudaFree(destr_parts);
    cudaFree(removed_nodes);
}


// assigns node to partition with maximum score
// kernel is too small, only k threads and 1 block
// either find a fix or serialize this
// threads are 95+% instruction inactive
__global__ void assignToBestPart(int k, float* results, int n, int* nodes, int* parts, int* int_costs, int* ext_costs, int* block_sums_i, int* block_sums_e, int blocks) {
    int tid = threadIdx.x;
    extern __shared__ int sdata[];
    if (tid < k) { // reduction for finding index of max value in results
        sdata[tid] = tid; //initialize sdata to partition ids
    }
    __syncthreads();
    int nextTid;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            nextTid = sdata[tid + stride];
            if (results[blockIdx.x * k + sdata[tid]] < results[blockIdx.x * k + nextTid]) {
                sdata[tid] = nextTid;
            }

        }
        __syncthreads();
    }
    if (tid == 0) {
        int partition = sdata[0];
        parts[nodes[blockIdx.x]] = partition; // assign node to index sdata[0]
        int final_sum_i = 0;
        int final_sum_e = 0;
        //printf("blockid %d gridim %d blocks %d\n", blockIdx.x, gridDim.x, blocks);
        for (int i = 0; i < blocks; i++) {
            final_sum_i += block_sums_i[blockIdx.x * k * blocks + partition * blocks + i];
            final_sum_e += block_sums_e[blockIdx.x * k * blocks + partition * blocks + i];
        }
        atomicAdd(&int_costs[partition], 2 * final_sum_i);
        atomicAdd(&ext_costs[partition], final_sum_e);
    }

}

void repair(int* parts, int k, int* destr_mask, int n, int destr_nodes, int m, int* int_costs, int* ext_costs, CSR* csr_rep, CSC* csc_rep) {
    //int i = 0;
    int node;
    float* d_result;
    cudaMalloc((void**)&d_result, k * destr_nodes * sizeof(float));
    float* result = (float*)malloc(k * destr_nodes * sizeof(float));
    int asgn;
    float temp_cost;
    int blocks = n / THREADS_PER_BLOCK + 1; // blocks/4 todo
    dim3 grid_dim(blocks, k, destr_nodes); // n/64 * k * m
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    int* block_sums_e, * block_sums_i;
    //printf("destr_nodes %d\n", destr_nodes);
    cudaMalloc((void**)&block_sums_e, blocks * k * destr_nodes * sizeof(int));
    cudaMalloc((void**)&block_sums_i, blocks * k * destr_nodes * sizeof(int));
    // loop is necessary because otherwise
    // nodes are added with incomplete information
    //for (int i = 0; i < (n * m / 100); i++) {
        //node = destr_mask[i];
    assignToParts << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (n, destr_mask, parts, int_costs, ext_costs, csr_rep, csc_rep, block_sums_i, block_sums_e, d_result);
    cudaDeviceSynchronize();
    assignToBestPart << <destr_nodes, k, 2 * k * sizeof(int) >> > (k, d_result, n, destr_mask, parts, int_costs, ext_costs, block_sums_i, block_sums_e, blocks);
    cudaDeviceSynchronize();
    //}
    free(result);
    cudaFree(d_result);
    cudaFree(block_sums_e);
    cudaFree(block_sums_i);
}


void lns_v0(int* in_parts, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR* row_rep, CSC* col_rep) {
    int* best = (int*)malloc(nodes_num * parts_num * sizeof(int));
    for (int i = 0; i < nodes_num; i++) {
        best[i] = in_parts[i];
    }
    //compute node costs
    int* d_temp_int_cost, * d_temp_ext_cost;

    int* int_cost = (int*)malloc(parts_num * sizeof(int));
    int* ext_cost = (int*)malloc(parts_num * sizeof(int));
    int* temp_int_cost = (int*)malloc(parts_num * sizeof(int));
    int* temp_ext_cost = (int*)malloc(parts_num * sizeof(int));
    cudaMalloc((void**)&d_temp_int_cost, parts_num * sizeof(int));
    cudaMalloc((void**)&d_temp_ext_cost, parts_num * sizeof(int));
    newComputeAllEdgeCost(best, row_rep, col_rep, parts_num, nodes_num, edges_num, int_cost, ext_cost);
    float best_cost = computeCost(int_cost, ext_cost, parts_num);
    float new_cost;
    int destr_nodes = nodes_num * m / 100;
    int* d_destr_mask, * temp;
    int* destr_mask = (int*)malloc(destr_nodes * sizeof(int));
    cudaMalloc((void**)&d_destr_mask, destr_nodes * sizeof(int));
    cudaMalloc((void**)&temp, nodes_num * sizeof(int));

    // copy CSR / CSC to device
    CSR* d_row_rep;
    CSC* d_col_rep;
    int* row_offsets, * col_offsets, * col_indexes, * row_indexes, * row_values, * col_values;
    printf("Allocation d_row, d_col\n");
    cudaMalloc((void**)&d_row_rep, sizeof(CSR));
    cudaMalloc((void**)&row_offsets, (nodes_num + 1) * sizeof(int));
    cudaMalloc((void**)&col_indexes, edges_num * sizeof(int));
    cudaMalloc((void**)&row_values, edges_num * sizeof(int));
    cudaMalloc((void**)&d_col_rep, sizeof(CSC));
    cudaMalloc((void**)&col_offsets, (nodes_num + 1) * sizeof(int));
    cudaMalloc((void**)&row_indexes, edges_num * sizeof(int));
    cudaMalloc((void**)&col_values, edges_num * sizeof(int));
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

    for (int iter = 0; iter < MAX_ITER; iter++) {
        printf("*****\nIteration %d start\n*****\n", iter);
        //reset values
        for (int i = 0; i < destr_nodes; i++) {
            destr_mask[i] = 0;
        }
        cudaMemcpy(temp, in_parts, nodes_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_int_cost, int_cost, parts_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_ext_cost, ext_cost, parts_num * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(temp_int_cost, d_temp_int_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_ext_cost, d_temp_ext_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);


        //destroy step
        //printf("Random generation start\n");
        computeRandomMask(destr_mask, nodes_num, m);
        //printf("Random generation end\n");
        cudaMemcpy(d_destr_mask, destr_mask, destr_nodes * sizeof(int), cudaMemcpyHostToDevice);
        //printf("Destroy start\n");
        
        destroy(temp, d_destr_mask, destr_nodes, parts_num, nodes_num, d_temp_int_cost, d_temp_ext_cost, 
                 d_row_rep, d_col_rep);
        //printf("Destroy end\n");
        cudaMemcpy(temp_int_cost, d_temp_int_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_ext_cost, d_temp_ext_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Cost after destroy %f \n", computeCost(temp_int_cost, temp_ext_cost, parts_num));


        //repair step
        //printf("Repair start\n");
        //repair(temp, parts_num, d_destr_mask, nodes_num, destr_nodes, m, d_temp_int_cost, d_temp_ext_cost, d_row_rep, d_col_rep);
        repair(temp, parts_num, d_destr_mask, nodes_num, destr_nodes, m, d_temp_int_cost, d_temp_ext_cost, 
                d_row_rep, d_col_rep);
        //printf("Repair end\n");

        //accept step
        cudaMemcpy(temp_int_cost, d_temp_int_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_ext_cost, d_temp_ext_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);


        if (checkMass(temp_int_cost, parts_num, max_mass)) {
            new_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num);
            printf("New cost found is: %f\n", new_cost);
            if (new_cost > best_cost)
                //printf("New best cost is: %f\n", new_cost);
                best_cost = new_cost;
            cudaMemcpy(best, temp, nodes_num * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    printf("Final cost is: %f\n", best_cost);

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
    cudaFree(d_row_rep);
    cudaFree(d_col_rep);
}