#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"
#include "../include/util.cuh"

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

__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__inline__ __device__ int warpReduceSum(int val) {
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
            warpReduce(sdata, threadIdx.x);
            warpReduce(sdata, threadIdx.x + blockDim.x);
        }

        // store block reduction result
        // this is needed because a single can have max 1024 threads
        // so if a node is connected to more than 1024 other nodes, multiple blocks are needed to handle it
        // as shared memory is only intra-block, global memory is needed

        if (threadIdx.x == 0) {
            atomicAdd(&block_sums_i[blockIdx.y * gridDim.x + blockIdx.x], sdata[0]);
            atomicAdd(&block_sums_e[blockIdx.y * gridDim.x + blockIdx.x], sdata[blockDim.x]);
        }
        __syncthreads();

        // only the first thread of the grid row is going to update the costs
        // atomic operation needed as multiple nodes might have been removed
        // from the same partition, so there could be simultaneous writes 
        if (ind == 0 && blockIdx.x == 0) {
            parts[node] = -1; // remove node from partition

            int final_sum_i = 0;
            int final_sum_e = 0;
            for (int i = 0; i < gridDim.x; i++) {
                final_sum_i += block_sums_i[blockIdx.y * gridDim.x + i];
                final_sum_e += block_sums_e[blockIdx.y * gridDim.x + i];
            }

            //update costs
            atomicSub(&int_costs[partition], final_sum_i);
            atomicSub(&ext_costs[partition], final_sum_e);
        }
    }
}

__global__ void removeNodes2(int* parts, int* nodes, int n, int* int_costs, int* ext_costs, 
                             int* r_offset, int* r_indexes, int* r_values, int* c_offset, int* c_indexes, int* c_values,
                             int* block_sums_i, int* block_sums_e, int* removed_nodes) {
    
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata_i[], sdata_e[];

    if (threadIdx.x == 0) {
        int node = nodes[blockIdx.z];
        sdata_i[0] = r_offset[node];
        sdata_i[1] = c_offset[node];
        sdata_i[2] = r_offset[node + 1] - sdata_i[0];
        sdata_i[3] = c_offset[node + 1] - sdata_i[1];
        sdata_i[4] = parts[node];
        if (ind == 0) {
            removed_nodes[node] = 1;
        }
    }
    __syncthreads();
    int start_r = sdata_i[0];
    int start_c = sdata_i[1];
    int r_size = sdata_i[2];
    int c_size = sdata_i[3];
    int k = sdata_i[4];
    // init of block sums
    // block_sums[k*gridDim.x...k*gridDim+block_id] is the reduction result of block block_id in row k
    // sdata init
    sdata_i[threadIdx.x] = 0;
    sdata_e[threadIdx.x] = 0;

    // gather values
    //int s_ind;
    __syncthreads();

    /*if (ind < r_size) {
        edge_node = r_indexes[start_r + ind];
        //printf("node %d edge_node %d s_ind %d\n", node, edge_node, s_ind);
        sdata[2*threadIdx.x + (parts[edge_node] != k)] = r_values[start_r + ind] * (1 + (parts[edge_node] == k) * (!removed_nodes[edge_node]));

    }*/
    for (int i = start_r + 4*ind ; i < r_size && i < start_r + 4*ind + 4; i++){
        if (parts[r_indexes[i]] == k){
            sdata_i[threadIdx.x] += r_values[i] * (1 + (!removed_nodes[r_indexes[i]]));
        } else {
            sdata_e[threadIdx.x] += r_values[i];
        }
    }
    for (int i = start_c + 4*ind - r_size; i < c_size && (i = start_c + 4*ind - r_size + 4); i++){
        if (parts[c_indexes[i]] == k){
            sdata_i[threadIdx.x] += c_values[i] * (1 + (!removed_nodes[c_indexes[i]]));
        } else {
            sdata_e[threadIdx.x] += c_values[i];
        }
    }
    /*
    else if (ind < r_size + c_size) {
        edge_node = c_indexes[start_c + ind - r_size];
        //printf("node %d edge_node %d s_ind %d\n", node, edge_node, s_ind);
        sdata[2*threadIdx.x + (parts[edge_node] != k)] = c_values[start_c + ind - r_size] * (1 + (parts[edge_node] == k) * (!removed_nodes[edge_node]));
    }*/
    __syncthreads();
    // reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < r_size + c_size) {
            sdata_i[threadIdx.x] += sdata_i[threadIdx.x + stride];
            sdata_e[threadIdx.x] += sdata_e[threadIdx.x + stride];
        }
        __syncthreads();
    } if (threadIdx.x < 32) {
        warpReduce(sdata_i, threadIdx.x);
        warpReduce(sdata_e, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //printf("gathered %d and %d\n", sdata[0], sdata[blockDim.x]);
        block_sums_i[blockIdx.z * gridDim.x + blockIdx.x] = sdata_i[0];
        block_sums_e[blockIdx.z * gridDim.x + blockIdx.x] = sdata_e[0];
    }
}
__global__ void removeNodes3(int* parts, int* nodes, int destr_nodes, int* int_costs, int* ext_costs, 
                            int* r_offset, int* r_indexes, int* r_values, int* c_offset, int* c_indexes, int* c_values,
                            int* removed_nodes) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < destr_nodes) {
        int node = nodes[ind];
        int k = parts[node];
        int start, end, size, edge_node, sum_i, sum_e;
        sum_i = 0;
        sum_e = 0;
        start = r_offset[node];
        end = r_offset[node+1];
        for (int i = start; i < end; i++){
            edge_node = r_indexes[i];
            if (parts[edge_node] == k){
                sum_i += (1 + !removed_nodes[edge_node]) * r_values[i];
            } else {
                sum_e += r_values[i];
            }
        }
        start = c_offset[node];
        end = c_offset[node+1];
        for (int i = start; i < end; i++){
            edge_node = c_indexes[i];
            if (parts[edge_node] == k){
                sum_i += (1 + !removed_nodes[edge_node]) * c_values[i];
            } else {
                sum_e += c_values[i];
            }
        }
        atomicSub(&int_costs[k], sum_i);
        atomicSub(&ext_costs[k], sum_e);
    }
}

__global__ void updatePartWeights(int* nodes, int* parts, int* out_i, int* out_e, int* costs_i, int* costs_e) {
    extern __shared__ int sdata[];

    sdata[threadIdx.x] = out_i[blockIdx.x * blockDim.x + threadIdx.x];
    sdata[threadIdx.x + blockDim.x] = out_e[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    } if (threadIdx.x < 32) {
        warpReduce(sdata, threadIdx.x);
        warpReduce(sdata, threadIdx.x + blockDim.x);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int partition = parts[nodes[blockIdx.x]];
        //printf("removing node %d from parts %d, with weights %d and %d\n", nodes[blockIdx.x], partition, sdata[0], sdata[blockDim.x]);
        parts[nodes[blockIdx.x]] = -1;
        atomicSub(&costs_i[partition], sdata[0]);
        atomicSub(&costs_e[partition], sdata[blockDim.x]);
    }
}
// Given k partitions and n*m/100 threads per block
// each threads check if the destr_mask[threadIdx.x] node is present in its block's 
// partition and destroys it if necessary
// usage should be destroy<<k, n*m/100>>
// costs update should be handled by another function

__global__ void getPartitionPerDestrNode(int* parts, int* destr_mask, int* destr_parts, int destr_nodes, int n) {
    int partition = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < destr_nodes) {
        int node = destr_mask[tid];
        int ind = partition * n + node;
        if (parts[ind] == 1) {
            destr_parts[tid] = partition;
        }
    }
}
// Assigns n*m/100 nodes to random partions

__device__ float computePartCost(float u, float ext) {
    return 100 * (u / (u + ext));
}

// uses 54 registers per thread, 27648 per block out of 65536
// should use around 130+ registers per thread for increase in performance
// 14.75% synch, 12.82% memory dep, 57.70% execution dep
// 10% branch divergence
__global__ void assignToParts(int n, int* nodes, int* parts, int* int_costs, int* ext_costs, CSR* csr_rep, CSC* csc_rep, int* block_sums_i, int* block_sums_e) {
    //should be a parallel reduction here
    int k = blockIdx.y;
    int block_id = blockIdx.x;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sdata[];

    int node = nodes[blockIdx.z];
    int start_r = csr_rep->offsets[node];
    int end_r = csr_rep->offsets[node + 1];
    int start_c = csc_rep->offsets[node];
    int end_c = csc_rep->offsets[node + 1];

    int r_size = end_r - start_r;
    int c_size = end_c - start_c;
    int max_size = r_size > c_size ? r_size : c_size;

    // init of block sums
    // block_sums[k*gridDim.x...k*gridDim+block_id] is the reduction result of block block_id in row k
    if (threadIdx.x == 0) {
        block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + block_id] = 0;
        block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + block_id] = 0;
    }
    // sdata init
    if (threadIdx.x < max_size) {
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
        edge_node = csc_rep->row_indexes[start_c + ind];
        s_ind = (parts[edge_node] != k) * blockDim.x; //used to know if value is be stored as ext or int
        sdata[threadIdx.x + s_ind] = csc_rep->values[start_c + ind];
    }
    // reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < max_size) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    } if (threadIdx.x < 32) {
        warpReduce(sdata, threadIdx.x);
        warpReduce(sdata, threadIdx.x + blockDim.x);
    }

    if (ind == 0) {
        atomicAdd(&block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + block_id], sdata[0]);
        atomicAdd(&block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + block_id], sdata[blockDim.x]);
    }

    // store final result
    // cambiare
    /*
    if (ind == 0 && block_id == 0) {
        int final_sum_i = 0;
        int final_sum_e = 0;
        for (int i = 0; i < gridDim.x; i++) { // fare reduction
            final_sum_i += block_sums_i[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + i];
            final_sum_e += block_sums_e[blockIdx.z * gridDim.y * gridDim.x + k * gridDim.x + i];
        }
        int mu_k = (int_costs[k] + final_sum_i);
        float old_cost = computePartCost(int_costs[k], ext_costs[k]);
        float new_cost = computePartCost(mu_k, ext_costs[k] + final_sum_e);
        result[blockIdx.z * gridDim.y + k] = new_cost - old_cost;
    }*/
}

__global__ void assignToParts2(int n, int* nodes, int* parts, int* int_costs, int* ext_costs, // util params
                               int *r_offset, int *r_indexes, int *r_values, int* c_offset, int* c_indexes, int* c_values, // graph
                               int *out_i, int *out_e) { // results
    int sum_i = 0;
    int sum_e = 0;
    extern __shared__ int sdata[];

    int partition = blockIdx.y;
    int res_ind = blockIdx.z * gridDim.y + blockIdx.y;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        int node = nodes[blockIdx.z];
        sdata[0] = r_offset[node];
        sdata[1] = c_offset[node];
        sdata[2] = r_offset[node + 1] - sdata[0];
        sdata[3] = c_offset[node + 1] - sdata[1];
    }
    __syncthreads();
    int start_r = sdata[0];
    int start_c = sdata[1];
    int r_size = sdata[2];
    int c_size = sdata[3];
    int edge_node;
    for (int i = ind; i < r_size; i += blockDim.x * gridDim.x) {
        //printf("thread %d block (%d,%d,%d) start_r %d i %d\n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z, start_r, i);
        edge_node = r_indexes[start_r + i];
        if (parts[edge_node] == partition) {
            sum_i += r_values[start_r + i];
        }
        else {
            sum_e += r_values[start_r + i];
        }
    }
    for (int i = ind; i < c_size; i += blockDim.x * gridDim.x) {
        edge_node = c_indexes[start_c + i];
        if (parts[edge_node] == partition) {
            sum_i += c_values[start_c + i];
        }
        else {
            sum_e += c_values[start_c + i];
        }
    }

    sum_i = warpReduceSum(sum_i);
    sum_e = warpReduceSum(sum_e);
    if ((threadIdx.x & 0x1f) == 0) {
        atomicAdd(&out_i[res_ind], sum_i);
        atomicAdd(&out_e[res_ind], sum_e);
    }
}

__device__ void assignToPart(int node, int part, int out_ind, int *parts, int start, int size, int *indexes, int *values,
                                                             int *out_i, int *out_e){
    int ind  = blockIdx.x * blockDim.x + threadIdx.x
    int edge_node;
    int sum_i = 0;
    int sum_e = 0;
    if (ind < size){
        edge_node = indexes[start_r + ind];
        if (parts[edges] == part){
            sum_i = values[start + ind];
        } else {
            sum_e = values[start + ind];
        }
    }
    sum_i = warpReduceSum(sum_i);
    sum_e = warpReduceSum(sum_e);
    if ((threadIdx.x & 0x1f) == 0) {
        atomicAdd(&out_i[out_ind], sum_i);
        atomicAdd(&out_e[out_ind], sum_e);
    }
}

// TODO
__global__ void assignToParts3(int n, int* nodes, int* parts, int* int_costs, int* ext_costs, // util params
                                int *r_offset, int *r_indexes, int *r_values, int* c_offset, int* c_indexes, int* c_values, // graph
                                int *out_i, int *out_e) { // results

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int node = nodes[ind];
    int start_r = r_offset[node];
    int end_r = r_offset[node+1];
    int start_c = c_offset[node];
    int end_c c_offset[node+1];
    int max_size = size_r < size_c ? size_c : size_r;
    int block_dim = min(1024, (max_size)/1024 + ((max_size%1024)>0));
    int gridx = destr_nodes/blockdim + (destr_nodes%blockdim > 0);
    dim3 grid_dim(gridx, k, 1);
    assignToPart<<<grid_dim, block_dim>>>(node, i, ind, parts, start_r, size_r, r_indexes, r_values,
                                        out_i, out_e);
    assignToPart<<<grid_dim, block_dim, 0, cudaStreamTailLaunch>>>(node, i, ind, parts, start_c, size_c, c_indexes, c_values,
                                        out_i, out_e);
}

__global__ void setToZero(int* arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = 0;
}
__global__ void setRemovedNodes(int* nodes, int* arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[nodes[tid]] = 1;
}


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
    removeNodes << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (parts, destr_mask, n, int_costs, ext_costs, row_rep, col_rep, block_sums_i, block_sums_e, removed_nodes);
    cudaDeviceSynchronize(); // probably not needed
    cudaFree(block_sums_i);
    cudaFree(block_sums_e);
    cudaFree(destr_parts);
    cudaFree(removed_nodes);
}

void destroy2(int* parts, int* destr_mask, int destr_nodes, int k, int n, int* int_costs, int* ext_costs,
              int* r_offset, int* r_indexes, int* r_values, int* c_offset, int* c_indexes, int* c_values) {
    int* block_sums_i, * block_sums_e, * destr_parts, * removed_nodes;
    cudaMalloc((void**)&block_sums_i, destr_nodes * BLOCKS_PER_ROW * sizeof(int));
    cudaMalloc((void**)&block_sums_e, destr_nodes * BLOCKS_PER_ROW * sizeof(int));
    cudaMalloc((void**)&destr_parts, destr_nodes * sizeof(int));
    // get partitions of destroyed nodes
    //getPartitionPerDestrNode << <dest_grid, THREADS_PER_BLOCK >> > (parts, destr_mask, destr_parts, destr_nodes, n);
    int sm_num; 
    cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, 0);
    int blockdim = min(1024, destr_nodes/sm_num);
    int gridx = destr_nodes/blockdim + (destr_nodes%blockdim > 0);
    dim3 grid_dim(gridx, 1, 1);
    dim3 block_dim(blockdim, 1, 1);
    cudaMalloc((void**)&removed_nodes, n * sizeof(int));
    // set to zero removed_nodes
    setToZero << <n / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (removed_nodes, n);
    cudaDeviceSynchronize();
    setRemovedNodes<<<destr_nodes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (destr_mask, removed_nodes, destr_nodes);
    cudaDeviceSynchronize();
    // remove nodes
    // removeNodes << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (parts, destr_mask, n, int_costs, ext_costs, row_rep, col_rep, block_sums_i, block_sums_e, removed_nodes);
    removeNodes3 << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (parts, destr_mask, destr_nodes, int_costs, ext_costs, 
                                                                                    r_offset, r_indexes, r_values,
                                                                                    c_offset, c_indexes, c_values,
                                                                                    removed_nodes);
    cudaDeviceSynchronize();
    //updatePartWeights << <destr_nodes, BLOCKS_PER_ROW, 2 * BLOCKS_PER_ROW * sizeof(int)>> > (destr_mask, parts, block_sums_i, block_sums_e, int_costs, ext_costs);
    //cudaDeviceSynchronize(); // probably not needed
    cudaFree(block_sums_i);
    cudaFree(block_sums_e);
    cudaFree(destr_parts);
    cudaFree(removed_nodes);
}

// assigns node to partition with maximum score
// kernel is too small, only k threads and 1 block
// either find a fix or serialize this
// threads are 95+% instruction inactive
__global__ void assignToBestPart(int k, float* results, int n, int* nodes, int* parts, int* int_costs, int* ext_costs, int* out_i, int* out_e) {
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
        /*
        int final_sum_i = 0;
        int final_sum_e = 0;
        //printf("blockid %d gridim %d blocks %d\n", blockIdx.x, gridDim.x, blocks);
        for (int i = 0; i < blocks; i++) {
            final_sum_i += block_sums_i[blockIdx.x * k * blocks + partition * blocks + i];
            final_sum_e += block_sums_e[blockIdx.x * k * blocks + partition * blocks + i];
        }*/
        atomicAdd(&int_costs[partition], 2 * out_i[blockIdx.x*k + partition]);
        atomicAdd(&ext_costs[partition], out_e[blockIdx.x * k + partition]);
    }

}

__global__ void gatherResults(int* out_i, int* out_e, int k, int* i_costs, int* e_costs, float* result, int n) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < n) {
        for (int i = 0; i < k; i++) {
            result[ind * k + i] = computePartCost(2 * out_i[ind * k + i] + i_costs[i], out_e[ind * k + i] + e_costs[i]) - computePartCost(i_costs[i], e_costs[i]);
        }
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
    assignToParts << <grid_dim, block_dim, 2 * THREADS_PER_BLOCK * sizeof(int) >> > (n, destr_mask, parts, int_costs, ext_costs, csr_rep, csc_rep, block_sums_i, block_sums_e);
    cudaDeviceSynchronize();
    //assignToBestPart << <destr_nodes, k, 2 * k * sizeof(int) >> > (k, d_result, n, destr_mask, parts, int_costs, ext_costs, block_sums_i, block_sums_e, blocks);
    cudaDeviceSynchronize();
    //}
    free(result);
    cudaFree(d_result);
    cudaFree(block_sums_e);
    cudaFree(block_sums_i);
}

void repair2(int* parts, int k, int* destr_mask, int n, int destr_nodes, int m, int* int_costs, int* ext_costs, 
             int* r_offset, int* r_indexes, int* r_values, int* c_offset, int* c_indexes, int* c_values) {
    //int i = 0;
    int node;
    float* d_result;
    int arr_size = k * destr_nodes;

    cudaMalloc((void**)&d_result, arr_size * sizeof(float));
    float* result = (float*)malloc(arr_size * sizeof(float));
    int blocks = n / THREADS_PER_BLOCK + 1; // blocks/4 todo
    dim3 grid_dim(blocks/4, k, destr_nodes); // n/64 * k * m
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    int* out_e, * out_i;
    //printf("destr_nodes %d\n", destr_nodes);
    cudaMalloc((void**)&out_e, arr_size * sizeof(int));
    cudaMalloc((void**)&out_i, arr_size * sizeof(int));
    // loop is necessary because otherwise
    // nodes are added with incomplete information
    //for (int i = 0; i < (n * m / 100); i++) {
        //node = destr_mask[i];
    setToZero<<<arr_size/1024 + 1, 1024>>>(out_e, arr_size);
    setToZero<<<arr_size/1024 + 1, 1024>>>(out_i, arr_size);
    cudaDeviceSynchronize();
    assignToParts2 << <grid_dim, block_dim, 4 * sizeof(int) >> > (n, destr_mask, parts, int_costs, ext_costs, 
                                                                                      r_offset, r_indexes, r_values,
                                                                                      c_offset, c_indexes, c_values,
                                                                                      out_i, out_e);
    cudaDeviceSynchronize();
    gatherResults << <destr_nodes/1024 + 1, 1024 >> > (out_i, out_e, k, int_costs, ext_costs, d_result, destr_nodes);
    cudaDeviceSynchronize();
    assignToBestPart << <destr_nodes, k, 2 * k * sizeof(int) >> > (k, d_result, n, destr_mask, parts, int_costs, ext_costs, out_i, out_e);
    cudaDeviceSynchronize();
    //}
    free(result);
    cudaFree(d_result);
    cudaFree(out_e);
    cudaFree(out_i);
}

__global__ void resetMask(int* mask, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        mask[tid] = 0;
    }
}

void lns_v1(int* in_parts, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR* row_rep, CSC* col_rep) {
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
        
        destroy2(temp, d_destr_mask, destr_nodes, parts_num, nodes_num, d_temp_int_cost, d_temp_ext_cost, 
                 row_offsets, col_indexes, row_values,
                 col_offsets, row_indexes, col_values);
        //printf("Destroy end\n");
        cudaMemcpy(temp_int_cost, d_temp_int_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_ext_cost, d_temp_ext_cost, parts_num * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Cost after destroy %f \n", computeCost(temp_int_cost, temp_ext_cost, parts_num));


        //repair step
        //printf("Repair start\n");
        //repair(temp, parts_num, d_destr_mask, nodes_num, destr_nodes, m, d_temp_int_cost, d_temp_ext_cost, d_row_rep, d_col_rep);
        repair2(temp, parts_num, d_destr_mask, nodes_num, destr_nodes, m, d_temp_int_cost, d_temp_ext_cost, 
                row_offsets, col_indexes, row_values,
                col_offsets, row_indexes, col_values);
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