#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"
#include "../include/util.cuh"

// legacy debug functions
void checkNodesPerPart(int *parts, int k, int n){
    int tot = 0;
    for (int i = 0; i < k*n; i++){
        tot += parts[i];
    }
    if (tot != n) printf("ERROR, counted %d nodes instead of %d.\n", tot, n);
    else printf("node check OK\n");
}

void checkPartsPerNode(int *parts, int k, int n){
    int res;
    for (int i = 0; i < n; i++){
        res = 0;
        for (int j = 0; j < k; j++){
            if (parts[j*n+i] == 1) res++;
        }
        if (res > 1) printf("Found node %d in multiple partitions\n", i);
    }
}

int *computeNodeCost(int *parts, int *weights, int parts_num, int nodes_num, int *costs){
    for (int i = 0; i < parts_num; i++) costs[i] = 0;

    for (int i = 0; i < parts_num; i++){
        for (int j = 0; j < nodes_num; j++){
            costs[i] += parts[i*nodes_num + j] * weights[j];
        }
    }
    return costs;
}

// legacy
void computeRandomAssignment(int * mask, int n, int m, int p){
    for (int i = 0; i < (n*m/100); i++){
        mask[i] = rand() % p;
    }
}

/*
Removes the destr_nodes nodes in destr_mask from the graph partitioning, represented by parts
int_costs and ext_costs are, respectively, the internal and external costs of the partitions
these should be temporary arrays tied to the current lns iteration
*/

void destroy(int *parts, int k, int *destr_mask, int n, int destr_nodes, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int ind;
    int node;
    for (int j = 0; j < destr_nodes; j++){
        node = destr_mask[j];        
        //printf("destroyed node %d\n", node);
        removeFromCost(parts, n, node, int_costs, ext_costs, csr_rep, csc_rep);
        //printf("updated costs\n");
    }
}

/*
Adds the destr_nodes nodes in destr_mask to the graph partitioning, represented by parts
int_costs and ext_costs are, respectively, the internal and external costs of the partitions
these should be temporary arrays tied to the current lns iteration
*/
void repair(int *parts, int *destr_mask, int n, int destr_nodes, int parts_num, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    //int i = 0;
    int k;
    int node;
    int best_k;
    float best_cost, temp_cost;
    int *temp_int_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_ext_cost = (int *)malloc(parts_num*sizeof(int));
    float old_cost = computeCost(int_costs, ext_costs, parts_num);
    
    for (int i = 0; i < destr_nodes; i++){
        node = destr_mask[i];
        best_cost = 0;
        best_k = 0;
        for (int j = 0; j < parts_num; j++){
            memcpy(temp_int_cost, int_costs, parts_num*sizeof(int));
            memcpy(temp_ext_cost, ext_costs, parts_num*sizeof(int));
            addToCost(parts, j, n, node, temp_int_cost, temp_ext_cost, csr_rep, csc_rep);
            temp_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num) - old_cost;
            
            if (temp_cost > best_cost){
                best_cost = temp_cost;
                best_k = j;
            }
        }
        //printf("adding node %d to part %d\n", node, best_k);
        parts[node] = best_k;
        addToCost(parts, best_k, n, node, int_costs, ext_costs, csr_rep, csc_rep);
    }
}

/*
Serial implementation of Large Neighbourhood Search
in_parts is a nodes_num-size array of values in {0...parts_num} representing the partition in which a node resides
max_mass and m are the function paraments for the maximum value of F(Si) and the percentage of nodes to remove at each iteration
row/col_rep are the CSR and CSC formats of the graph, where the values are the edge weights
*/
void lns_serial(int *in_parts, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR *row_rep, CSC *col_rep){
    int *best = (int *) malloc(nodes_num*sizeof(int));
    for (int i = 0; i < nodes_num; i++){
        best[i] = in_parts[i];
    }
    //compute node costs
    int *int_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_int_cost = (int *)malloc(parts_num*sizeof(int));
    //computeNodeCost(best, weights, parts_num, nodes_num, node_cost);
    //compute edge costs
    int *ext_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_ext_cost = (int *)malloc(parts_num*sizeof(int));
    newComputeAllEdgeCost(best, row_rep, col_rep, parts_num, nodes_num, edges_num, int_cost, ext_cost);
    float best_cost = computeCost(int_cost, ext_cost, parts_num);
    float new_cost;
    int destr_nodes = nodes_num*m/100;
    int *destr_mask = (int *)malloc(destr_nodes*sizeof(int));
    int *temp = (int *) malloc(nodes_num*sizeof(int));
    
    srand(time(NULL));

    printf("Initial cost is: %f\n", best_cost);

    for (int iter = 0; iter < MAX_ITER; iter++){
        printf("Iteration %d start\n", iter);
        //reset values
        for (int i = 0; i < destr_nodes; i++){
            destr_mask[i] = 0;
        }
        memcpy(temp, in_parts, nodes_num*sizeof(int));
        memcpy(temp_int_cost, int_cost, parts_num*sizeof(int));
        memcpy(temp_ext_cost, ext_cost, parts_num*sizeof(int));

        //destroy step
        computeRandomMask(destr_mask, nodes_num, m);
        destroy(temp, parts_num, destr_mask, nodes_num, destr_nodes, temp_int_cost, temp_ext_cost, row_rep, col_rep);
        printf("cost after destroy: %f\n", computeCost(temp_int_cost, temp_ext_cost, parts_num));
        //repair step
        repair(temp, destr_mask, nodes_num, destr_nodes, parts_num, temp_int_cost, temp_ext_cost, row_rep, col_rep);

        //accept step
        if (checkMass(int_cost, parts_num, max_mass)){
            new_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num);
            if (new_cost > best_cost)
            printf("New best cost is: %f\n", new_cost);
                best_cost = new_cost;
                memcpy(best, temp, nodes_num*sizeof(int));
        }
        //debug only
        //checkNodesPerPart(temp, parts_num, nodes_num);
        //checkPartsPerNode(temp, parts_num, nodes_num);
    }
    printf("Final cost is: %f\n", best_cost);
    free(destr_mask);
    free(temp);
    free(int_cost);
    free(temp_int_cost);
    free(ext_cost);
    free(temp_ext_cost);
}