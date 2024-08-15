#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "lns.cuh" // CSR/CSC structs, defines
#include "init.cuh" // init functions


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

int computeMass(int ind, int *parts, int nodes_num, int *weights){
    int tot_mass = 0;
    for (int i = 0; i < nodes_num; i++){
        tot_mass += parts[ind*nodes_num+i]*weights[i];
    }
    return tot_mass;
}

int checkMass(int *parts, int *weights, int parts_num, int nodes_num, int max_mass){
    for (int i = 0; i < parts_num; i++){
        if (computeMass(i, parts, nodes_num, weights) > max_mass)
            return 0;
    }
    return 1;
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

int computeEdgeCost(int *parts, int part_id, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num){
    int ind = 0;
    int start, end;
    int res = 0;
    for (int i = 0; i < nodes_num; i++){
        ind = part_id*nodes_num+i;
        if (parts[ind] == 1){
            // out edges
            start = row_rep -> offsets[i];
            end = row_rep -> offsets[i+1];
            for (int j = start; j < end; j++){
                res += row_rep -> values[j];
            }
            // in edges
            /*
            start = col_rep -> offsets[i];
            end = col_rep -> offsets[i+1];
            for (int j = start; j < end; j++){
                res += col_rep -> values[j];
            }
            */
        }
    }
    return res;
}

void computeAllEdgeCost(int *parts, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *costs){
    for (int i = 0; i < parts_num; i++){
        costs[i] = computeEdgeCost(parts, i, row_rep, col_rep, parts_num, nodes_num, edges_num);
    }
}

// Random functions

void computeRandomMask(int * mask, int n, int m){
    int ind;
    for (int i = 0; i < (n*m/100); i++){
        ind = rand() % n;
        if (mask[ind] == 1) i=i-1;
        else mask[ind] = 1;
    }
}

void computeRandomAssignment(int * mask, int n, int m, int p){
    for (int i = 0; i < (n*m/100); i++){
        mask[i] = rand() % p;
    }
}

int removeFromCost(int *parts, int k, int n, int node, int *costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only remove cost of edges going in/out of the partition
            costs[k] -= csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going into the partition
            costs[k] -= csc_rep -> values[z];
        }
    }
    return res;
}

int addToCost(int *parts, int k, int n, int node, int *costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going out of the partition
            costs[k] += csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going into the partition
            costs[k] += csc_rep -> values[z];
        }
    }
    return res;
}

void destroy(int *parts, int k, int *destr_mask, int n, int *weights, int *node_costs, int *edge_costs, CSR *csr_rep, CSC *csc_rep){
    int ind;
    for (int i = 0; i < k; i++){
        for (int j = 0; j < n; j++){
            ind = i*n+j;
            if (destr_mask[j] == 1 && parts[ind] == destr_mask[j]){
                parts[ind] = 0;
                printf("destroyed node %d from part %d\n", j, i);
                edge_costs[k] -= removeFromCost(parts, k, n, j, edge_costs, csr_rep, csc_rep);
                node_costs[k] -= weights[j];
                printf("updated costs\n");
            }
        }
    }
}

void repair(int *parts, int k, int *destr_mask, int *asgn_mask, int n, int *weights, int *node_costs, int *edge_costs, CSR *csr_rep, CSC *csc_rep){
    int i = 0;
    for (int j = 0; j < n; j++){
        if (destr_mask[j] == 1){
            parts[asgn_mask[i]*n+j] = 1;
            edge_costs[i] -= addToCost(parts, i, n, j, edge_costs, csr_rep, csc_rep);
            node_costs[i] -= weights[j];
            i++;
        }
    }
}

int computeCost(int *node_costs, int *edge_costs, int k){
    int res = 0;
    for (int i = 0; i < k; i++){
        res += 2*node_costs[i] / (2*node_costs[i] * edge_costs[i]); 
    }
    return res;
}

void lns(int *in_parts, int *weights, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR *row_rep, CSC *col_rep){
    int *best = (int *) malloc(nodes_num*parts_num*sizeof(int));
    for (int i = 0; i < nodes_num*parts_num; i++){
        best[i] = in_parts[i];
    }
    //compute node costs
    int *node_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_node_cost = (int *)malloc(parts_num*sizeof(int));
    computeNodeCost(best, weights, parts_num, nodes_num, node_cost);
    //compute edge costs
    int *edge_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_edge_cost = (int *)malloc(parts_num*sizeof(int));
    computeAllEdgeCost(best, row_rep, col_rep, parts_num, nodes_num, edges_num, edge_cost);
    for (int i = 0; i < parts_num; i++){
        printf("init node cost %d \ninit edge cost %d\n", node_cost[i], edge_cost[i]);
    }
    int best_cost = computeCost(node_cost, edge_cost, parts_num);
    int new_cost;
    int *destr_mask = (int *)malloc(nodes_num*sizeof(int));
    int *asgn_mask = (int *)malloc((nodes_num*m/100)*sizeof(int));
    int *temp = (int *) malloc(nodes_num*parts_num*sizeof(int));
    srand(time(NULL));

    for (int iter = 0; iter < MAX_ITER; iter++){
        printf("Iteration %d start\n", iter);
        //reset values
        for (int i = 0; i < nodes_num; i++){
            destr_mask[i] = 0;
        }
        memcpy(temp, in_parts, nodes_num*parts_num*sizeof(int));
        memcpy(temp_node_cost, node_cost, parts_num*sizeof(int));
        memcpy(temp_edge_cost, edge_cost, parts_num*sizeof(int));

        printf("Destroy step %d\n", iter);
        //destroy step
        computeRandomMask(destr_mask, nodes_num, m);
        destroy(temp, parts_num, destr_mask, nodes_num, weights, temp_node_cost, temp_edge_cost, row_rep, col_rep);

        printf("Repair step %d\n", iter);
        //repair step
        computeRandomAssignment(asgn_mask, nodes_num, m, parts_num);
        repair(temp, parts_num, destr_mask, asgn_mask, nodes_num, weights, temp_node_cost, temp_edge_cost, row_rep, col_rep);

        printf("Accept step %d\n", iter);
        //accept step
        if (checkMass(temp, weights, parts_num, nodes_num, max_mass)){
            new_cost = computeCost(temp_node_cost, temp_edge_cost, parts_num);
            if (new_cost > best_cost)
                memcpy(best, temp, nodes_num*parts_num*sizeof(int));
        }
        //debug only
        //checkNodesPerPart(temp, parts_num, nodes_num);
        //checkPartsPerNode(temp, parts_num, nodes_num);
    }
    free(destr_mask);
}

void csrTest(int *offsets, int *columns, int *values, int n, int e){
    // csr test results
    int   h_csr_offsets[]  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int   h_csr_columns[]  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int h_csr_values[]   = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int   h_csr_offsets_result[]  = { 0, 2, 5, 7, 9, 9, 9, 11, 11, 13, 14 };
    int   h_csr_columns_result[]  = { 4, 5, 4, 5, 6, 3, 4, 7, 8, 7, 9, 7, 9, 7 };
    int h_csr_values_result[]   = { 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1};
    int correct = 1;
    printf("Testing...\n");

    for (int i = 0; i < n + 1; i++) {
        if (h_csr_offsets[i] != h_csr_offsets_result[i]) {
            correct = 0;break;
        }
    }
    if (correct)
        printf("offset test PASSED\n");
    else
        printf("offset test FAILED: wrong result\n");
    correct = 1;

    for (int i = 0; i < e; i++) {
        if (h_csr_columns[i] != h_csr_columns_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("columns test PASSED\n");
    else
        printf("columns test FAILED: wrong result\n");
    correct = 1;

    for (int i = 0; i < e; i++) {
        if (h_csr_values[i] != h_csr_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("values test PASSED\n");
    else
        printf("values test FAILED: wrong result\n");

}

int main(){
    int nodes_num, edges_num, parts_num;
    printf("Reading input...\n");
    FILE *in_file  = fopen("graph.txt", "r");
    char line[100];
    if (in_file == NULL){  
              printf("Error! Could not open file\n");
              exit(-1); // must include stdlib.h
            }
    // read number of nodes, edges and partitions
    fgets(line, 100, in_file);
    sscanf(line, "%d %d %d", &nodes_num, &edges_num, &parts_num);
    // init structures
    int *weights = (int *) malloc(nodes_num*sizeof(int));
    int *parts = (int *) malloc(nodes_num*sizeof(int));
    int *partitions = (int *) malloc(parts_num*nodes_num*sizeof(int));
    int *mat = (int *) malloc(nodes_num*nodes_num*sizeof(int));
    // read rest of the input
    readInput(in_file, partitions, weights, parts, nodes_num, edges_num, parts_num, mat);

    // setup csr representation
    int *h_csr_offsets = (int *) malloc((nodes_num + 1) * sizeof(int));
    int *h_csr_columns = (int *) malloc(edges_num * sizeof(int));
    int *h_csr_values = (int *) malloc(edges_num * sizeof(int));

    csrSetup(nodes_num, edges_num, mat, h_csr_offsets, h_csr_columns, h_csr_values);

    // setup csc representation
    int *h_csc_offsets = (int *) malloc((nodes_num + 1) * sizeof(int));
    int *h_csc_columns = (int *) malloc(edges_num * sizeof(int));
    int *h_csc_values = (int *) malloc(edges_num * sizeof(int));

    cscSetup(nodes_num, edges_num, mat, h_csc_offsets, h_csc_columns, h_csc_values);


    // generate csr rep
    // Device memory management

    //csrTest(h_csr_offsets, h_csr_columns, h_csr_values, edges_num, nodes_num);

    CSR *row_rep = (CSR*) malloc(sizeof(CSR));
    row_rep -> offsets = h_csr_offsets;
    row_rep -> col_indexes = h_csr_columns;
    row_rep -> values = h_csr_values;

    CSC *col_rep = (CSC*) malloc(sizeof(CSC));
    col_rep -> offsets = h_csc_offsets;
    col_rep -> row_indexes = h_csc_columns;
    col_rep -> values = h_csc_values;

    lns(partitions, weights, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);

    free(partitions);
    free(weights);
    free(parts);
    return 1;
}