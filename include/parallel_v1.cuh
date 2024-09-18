void lns_v1(int *in_parts, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR *row_rep, CSC *col_rep);

__global__ void resetMask(int* mask, int size);
__global__ void gatherResults(int* out_i, int* out_e, int k, int* i_costs, int* e_costs, float* result, int n);
__global__ void setRemovedNodes(int* nodes, int* arr, int n);
__global__ void setToZero(int* arr, int n);
__global__ void getPartitionPerDestrNode(int* parts, int* destr_mask, int* destr_parts, int destr_nodes, int n);
__global__ void updatePartWeights(int* nodes, int* parts, int* out_i, int* out_e, int* costs_i, int* costs_e);