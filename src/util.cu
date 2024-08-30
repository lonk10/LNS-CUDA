#include <stdio.h>
#include "../include/lns.cuh"


int computeMass(int ind, int *parts, int nodes_num, int *weights){
    int tot_mass = 0;
    for (int i = 0; i < nodes_num; i++){
        tot_mass += parts[ind*nodes_num+i]*weights[i];
    }
    return tot_mass;
}

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
    for (int i = 0; i < k; i++){
        u = (float) (int_costs[i]);
        //printf("res %d:%f ", i, 100*(u/ (u+(float)ext_costs[i])));
        res += 100 * (u / (u + (float) ext_costs[i])); 
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

void computeAllEdgeCost(int *parts, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_costs, int *ext_costs){
    for (int i = 0; i < parts_num; i++){
        computeEdgeCost(parts, i, row_rep, col_rep, parts_num, nodes_num, edges_num, int_costs, ext_costs);
    }
}

// Random functions


void computeRandomMask(int * mask, int n, int m){
    int i = 0;
    int max = n*m/100;
    int *check = (int*) malloc(n*sizeof(int));
    int rand_node;
    for (int j = 0; j < n; j++){
        check[j] = 0;
    }
    while (i < max){
        rand_node = rand() % n;
        if (check[rand_node] == 0){
            mask[i] = rand_node;
            check[rand_node] = 1;
            i++;
        }
    }
    free(check);
}

// Removes costs tied to node n in partition k
void removeFromCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    int edge_node;
    for (int z = start; z < end; z++){
        edge_node = csr_rep -> col_indexes[z];
        if (parts[k*n+edge_node] == 0){ // only remove cost of edges going in/out of the partition
            ext_costs[k] -= csr_rep -> values[z];
        } else {
            int_costs[k] -= csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        edge_node = csc_rep -> row_indexes[z];
        if (parts[k*n+edge_node] == 0){ // only add cost of edges going into the partition
            ext_costs[k] -= csc_rep -> values[z];
        } else {
            int_costs[k] -= csc_rep -> values[z];
        }
    }
}

// Adds costs tied to node n in partition k
int addToCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    int edge_node;
    for (int z = start; z < end; z++){
        edge_node = csr_rep -> col_indexes[z];
        if (parts[k*n+edge_node] == 0){ // only add cost of edges going out of the partition
            ext_costs[k] += csr_rep -> values[z];
        } else {
            int_costs[k] += csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        edge_node = csc_rep -> row_indexes[z];
        if (parts[k*n+edge_node] == 0){ // only add cost of edges going into the partition
            ext_costs[k] += csc_rep -> values[z];
        } else {
            int_costs[k] += csc_rep -> values[z];
        }
    }
    return res;
}