#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"




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
                if (!parts[node]) {
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
                if (!parts[node]) {
                    int_res += col_rep -> values[j];
                } else {
                    ext_res += col_rep -> values[j];
                }
            }
        }
    }
    *int_cost = int_res;
    *ext_cost = ext_res;
}

void computeAllEdgeCost(int *parts, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_costs, int *ext_costs){
    for (int i = 0; i < parts_num; i++){
        computeEdgeCost(parts, i, row_rep, col_rep, parts_num, nodes_num, edges_num, &int_costs[i], &ext_costs[i]);
    }
}

// Random functions

// Stores n*m/100 UNIQUE values in mask
void computeRandomMask(int * mask, int n, int m){
    int i = 0;
    int max = n*m/100;
    int *check = (int*) malloc(n*sizeof(int));
    int rand_node;
    for (int j = 0; j < n; j++){    // init
        check[j] = 0;
    }
    while (i < max){
        rand_node = rand() % n;
        if (check[rand_node] == 0){ // check if node has already been generated
            mask[i] = rand_node;
            check[rand_node] = 1;   // set node to generated
            i++;
        }
    }
    free(check);
}

void computeRandomAssignment(int * mask, int n, int m, int p){
    for (int i = 0; i < (n*m/100); i++){
        mask[i] = rand() % p;
    }
}

void removeFromCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only remove cost of edges going in/out of the partition
            ext_costs[k] -= csr_rep -> values[z];
        } else {
            int_costs[k] -= csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going into the partition
            ext_costs[k] -= csc_rep -> values[z];
        } else {
            int_costs[k] -= csc_rep -> values[z];
        }
    }
}

int addToCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int res = 0;
    int start = csr_rep -> offsets[node];
    int end = csr_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going out of the partition
            ext_costs[k] += csr_rep -> values[z];
        } else {
            int_costs[k] += csr_rep -> values[z];
        }
    }
    start = csc_rep -> offsets[node];
    end = csc_rep -> offsets[node+1];
    for (int z = start; z < end; z++){
        if (parts[k*n+z] == 0){ // only add cost of edges going into the partition
            ext_costs[k] += csc_rep -> values[z];
        } else {
            int_costs[k] += csc_rep -> values[z];
        }
    }
    return res;
}

void destroy(int *parts, int k, int *destr_mask, int n, int m, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    int ind;
    /*
    for (int i = 0; i < k; i++){
        for (int j = 0; j < n; j++){
            ind = i*n+j;
            if (destr_mask[j] == 1 && parts[ind] == destr_mask[j]){
                parts[ind] = 0;
                //printf("destroyed node %d from part %d\n", j, i);
                edge_costs[k] -= removeFromCost(parts, k, n, j, edge_costs, csr_rep, csc_rep);
                node_costs[k] -= weights[j];
                //printf("updated costs\n");
            }
        }
    }*/
    int node;
    for (int i = 0; i < k; i++){
        for (int j = 0; j < (n*m/100); j++){
            node = destr_mask[j];
            ind = i*n+node;
            if (parts[ind] == 1){
                parts[ind] = 0;
                //printf("destroyed node %d from part %d\n", j, i);
                removeFromCost(parts, k, n, node, int_costs, ext_costs, csr_rep, csc_rep);
                //printf("updated costs\n");
            }
        }
    }
}

void repair(int *parts, int *destr_mask, int *asgn_mask, int n, int m, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep){
    //int i = 0;
    int k;
    int node;
    for (int i = 0; i < (n*m/100); i++){
        k = asgn_mask[i];
        node = destr_mask[i];
        parts[k*n+node] = 1;
        //printf("added node %d to part %d\n", j, k);
        addToCost(parts, k, n, node, int_costs, ext_costs, csr_rep, csc_rep);
    }
}

float computeCost(int *int_costs, int *ext_costs, int k){
    float res = 0;
    float u = 0;
    for (int i = 0; i < k; i++){
        u = (float) 2*(int_costs[i]);
        printf("%f / (%f + %d = %f) = %f\n", u, u, ext_costs[i], (u+(float)ext_costs[i]), (u/ (u+(float)ext_costs[i])));
        res += u / (u + (float) ext_costs[i]); 
    }
    return res;
}

void lns(int *in_parts, int *weights, int parts_num, int nodes_num, int edges_num, int max_mass, int m, CSR *row_rep, CSC *col_rep){
    int *best = (int *) malloc(nodes_num*parts_num*sizeof(int));
    for (int i = 0; i < nodes_num*parts_num; i++){
        best[i] = in_parts[i];
    }
    //compute node costs
    int *int_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_int_cost = (int *)malloc(parts_num*sizeof(int));
    //computeNodeCost(best, weights, parts_num, nodes_num, node_cost);
    //compute edge costs
    int *ext_cost = (int *)malloc(parts_num*sizeof(int));
    int *temp_ext_cost = (int *)malloc(parts_num*sizeof(int));
    computeAllEdgeCost(best, row_rep, col_rep, parts_num, nodes_num, edges_num, int_cost, ext_cost);
    for (int i = 0; i < parts_num; i++){
        printf("init node cost %d \ninit edge cost %d\n", int_cost[i], ext_cost[i]);
    }
    float best_cost = computeCost(int_cost, ext_cost, parts_num);
    float new_cost;
    int destr_nodes = nodes_num*m/100;
    int *destr_mask = (int *)malloc(destr_nodes*sizeof(int));
    int *asgn_mask = (int *)malloc(destr_nodes*sizeof(int));
    int *temp = (int *) malloc(nodes_num*parts_num*sizeof(int));
    srand(time(NULL));

    printf("Initial cost is: %f\n", best_cost);

    for (int iter = 0; iter < MAX_ITER; iter++){
        //printf("Iteration %d start\n", iter);
        //reset values
        for (int i = 0; i < destr_nodes; i++){
            destr_mask[i] = 0;
        }
        memcpy(temp, in_parts, nodes_num*parts_num*sizeof(int));
        memcpy(temp_int_cost, int_cost, parts_num*sizeof(int));
        memcpy(temp_ext_cost, ext_cost, parts_num*sizeof(int));

        //printf("Destroy step %d\n", iter);
        //destroy step
        computeRandomMask(destr_mask, nodes_num, m);
        destroy(temp, parts_num, destr_mask, nodes_num, m, temp_int_cost, temp_ext_cost, row_rep, col_rep);

        //printf("Repair step %d\n", iter);
        //repair step
        computeRandomAssignment(asgn_mask, nodes_num, m, parts_num);
        repair(temp, destr_mask, asgn_mask, nodes_num, m, temp_int_cost, temp_ext_cost, row_rep, col_rep);

        //printf("Accept step %d\n", iter);
        //accept step
        if (checkMass(temp, weights, parts_num, nodes_num, max_mass)){
            new_cost = computeCost(temp_int_cost, temp_ext_cost, parts_num);
            if (new_cost > best_cost)
            printf("New best cost is: %f\n", new_cost);
                best_cost = new_cost;
                memcpy(best, temp, nodes_num*parts_num*sizeof(int));
        }
        //debug only
        //checkNodesPerPart(temp, parts_num, nodes_num);
        //checkPartsPerNode(temp, parts_num, nodes_num);
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
    free(destr_mask);
    //printf("snapp:\n");
    free(asgn_mask);
    //printf("snoop:\n");
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
    int *h_csc_rows = (int *) malloc(edges_num * sizeof(int));
    int *h_csc_values = (int *) malloc(edges_num * sizeof(int));

    cscSetup(nodes_num, edges_num, mat, h_csc_offsets, h_csc_rows, h_csc_values);


    // generate csr rep
    // Device memory management

    //csrTest(h_csr_offsets, h_csr_columns, h_csr_values, nodes_num, edges_num);
    //cscTest(h_csc_offsets, h_csc_rows, h_csc_values, nodes_num, edges_num);

    CSR *row_rep = (CSR*) malloc(sizeof(CSR));
    row_rep -> offsets = h_csr_offsets;
    row_rep -> col_indexes = h_csr_columns;
    row_rep -> values = h_csr_values;

    CSC *col_rep = (CSC*) malloc(sizeof(CSC));
    col_rep -> offsets = h_csc_offsets;
    col_rep -> row_indexes = h_csc_rows;
    col_rep -> values = h_csc_values;

    lns(partitions, weights, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);

    free(partitions);
    free(weights);
    free(parts);
    return 1;
}