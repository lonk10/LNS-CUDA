#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "./include/lns.cuh"
#include "./include/init.cuh"
#include "./include/util.cuh"
#include "./include/serial.cuh"
#include "./include/parallel_v1.cuh"


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

    printf("###################################\n");
    printf("#### STARTING SERIAL EXECUTION ####\n");
    printf("###################################\n");
    lns_serial(partitions, weights, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);

    printf("########################################\n");
    printf("#### STARTING PARALLEL_V1 EXECUTION ####\n");
    printf("########################################\n");
    lns_v1(partitions, weights, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);
    

    free(partitions);
    free(weights);
    free(parts);
    return 1;
}