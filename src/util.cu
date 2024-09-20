#include <stdio.h>
#include "../include/lns.cuh"
#include <stdlib.h>


int computeMass(int ind, int *parts, int nodes_num, int *weights){
    int tot_mass = 0;
    for (int i = 0; i < nodes_num; i++){
        tot_mass += parts[ind*nodes_num+i]*weights[i];
    }
    return tot_mass;
}

// Checks that forall partitions Si, int_costs[Si] < max_mass
int checkMass(int *int_costs, int parts_num, int max_mass){
    for (int i = 0; i < parts_num; i++){
        if (int_costs[i] > max_mass){
            printf("mass check not passed, temp int cost[%d] is %d\n", i, int_costs[i]);
            return 0;
        }
        //printf("temp int cost[%d] is %d\n", i, int_costs[i]);
    }
    return 1;
}

float computeCost(int *int_costs, int *ext_costs, int k){
    float res = 0;
    float u = 0;
    float n;
    for (int i = 0; i < k; i++){
        u = (float) (int_costs[i]);
        //printf("res %d:%f ", i, 100*(u/ (u+(float)ext_costs[i])));
        n = (u / (u + (float) ext_costs[i]));
        if (!isnan(n)) res += 100 * n; 
    }
    //printf("\n");
    return res;
}


void computeEdgeCost(int *parts, int part_id, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_cost, int *ext_cost){
    int ind = 0;
    int start, end;
    int int_res = 0;
    int ext_res = 0;
    int node;
    for (int i = 0; i < nodes_num; i++){
        ind = part_id*nodes_num+i;
        if (parts[ind]){
            // out edges
            start = row_rep -> offsets[i];
            end = row_rep -> offsets[i+1];
            for (int j = start; j < end; j++){
                node = row_rep -> col_indexes[j];
                if (parts[part_id*nodes_num+node]) {
                    int_res += row_rep -> values[j];
                } else {
                    ext_res += row_rep -> values[j];
                }
            }
            // in edges
            
            start = col_rep -> offsets[i];
            end = col_rep -> offsets[i+1];
            for (int j = start; j < end; j++){
                node = col_rep -> row_indexes[j];
                if (parts[part_id*nodes_num+node]) {
                    int_res += col_rep -> values[j];
                } else {
                    ext_res += col_rep -> values[j];
                }
            }
        }
    }
    int_cost[part_id] = int_res;
    ext_cost[part_id] = ext_res;
}

// legacy function
void computeAllEdgeCost(int *parts, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_costs, int *ext_costs){
    for (int i = 0; i < parts_num; i++){
        computeEdgeCost(parts, i, row_rep, col_rep, parts_num, nodes_num, edges_num, int_costs, ext_costs);
    }
}


// computes the internal (int_costs) and external (ext_costs) costs of all partitions (parts)
void newComputeAllEdgeCost(int* parts, CSR* row_rep, CSC* col_rep, int parts_num, int nodes_num, int edges_num, int* int_costs, int* ext_costs) {
    for (int i = 0; i < parts_num; i++) {
        int_costs[i] = 0;
        ext_costs[i] = 0;
    }
    int start, end;
    int int_res = 0;
    int ext_res = 0;
    int partition;
    int node;
    for (int i = 0; i < nodes_num; i++) {
        int_res = 0;
        ext_res = 0;
        partition = parts[i];
        start = row_rep->offsets[i];
        end = row_rep->offsets[i + 1];
        for (int j = start; j < end; j++) {
            node = row_rep->col_indexes[j];
            if (parts[node] == parts[i]) {
                int_res += row_rep->values[j];
            }
            else {
                ext_res += row_rep->values[j];
            }
        }

        start = col_rep->offsets[i];
        end = col_rep->offsets[i + 1];
        for (int j = start; j < end; j++) {
            node = col_rep->row_indexes[j];
            if (parts[node] == parts[i]) {
                int_res += col_rep->values[j];
            }
            else {
                ext_res += col_rep->values[j];
            }
        }
        //printf("adding %d and %d to part %d\n", int_res, ext_res, partition);
        int_costs[partition] += int_res;
        ext_costs[partition] += ext_res;
    }
}

// Random functions


// Computes a random mask of m unique values in the range 0..n
void computeRandomMask(int* mask, int n, int m) {
    int i = 0;
    int max = n * m / 100;
    int rand_node;
    unsigned char *is_used = (unsigned char *) malloc(n*sizeof(unsigned char)); // flags
    for (int z = 0; z < n; z++) {
        is_used[z] = 0;
    }

    int j = 0;
    int rn, rm;
    for (int i = n - max; i < n && j < max; i++){
        rand_node = rand() % (i+1);
        if (is_used[rand_node]) rand_node = i;
        mask[j++] = rand_node;
        is_used[rand_node] = 1;
    }
    free(is_used);
}

// Removes costs tied to node n in partition k
void removeFromCost(int *parts, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    int edge_node;
    int k = parts[node];
    int sum_i = 0;
    int sum_e = 0;
    for (int z = start; z < end; z++){
        edge_node = csr_rep -> col_indexes[z];
        if (parts[edge_node] == k){ // only remove cost of edges going in/out of the partition
            sum_i += 2 * csr_rep -> values[z];
        } else {
            sum_e += csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        edge_node = csc_rep -> row_indexes[z];
        if (parts[edge_node] == k){ // only add cost of edges going into the partition
            sum_i += 2 * csc_rep -> values[z];
        } else {
            sum_e += csc_rep -> values[z];
        }
    }
    int_costs[k] -= sum_i;
    ext_costs[k] -= sum_e;
}

// Adds costs tied to node n in partition k
int addToCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    int edge_node;
    for (int z = start; z < end; z++){
        edge_node = csr_rep -> col_indexes[z];
        if (parts[edge_node] == k){ // only add cost of edges going out of the partition
            int_costs[k] += 2 * csr_rep -> values[z];
        } else {
            ext_costs[k] += csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        edge_node = csc_rep -> row_indexes[z];
        if (parts[edge_node] == k){ // only add cost of edges going into the partition
            int_costs[k] += 2 * csc_rep -> values[z];
        } else {
            ext_costs[k] += csc_rep -> values[z];
        }
    }
    return res;
}