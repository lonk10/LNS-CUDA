#include <cuda.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "./include/lns.cuh"
#include "./include/init.cuh"
#include "./include/util.cuh"
#include "./include/serial.cuh"
#include "./include/parallel_v1.cuh"


int main(int argc, char* argv[]){
    int nodes_num, edges_num, parts_num;
    printf("Reading input...\n");

    char *file_name;
    if (argc > 1){
      file_name = argv[1];
    } else{
      file_name = "./graph.txt";
    }
    //FILE *in_file  = fopen("./data/guangzhou.graph", "r");
    FILE *in_file  = fopen(file_name, "r");
    char line[100];
    if (in_file == NULL){  
              printf("Error! Could not open file\n");
              exit(-1); // must include stdlib.h
            }
    // read number of nodes, edges and partitions
    fgets(line, 100, in_file);
    sscanf(line, "%d %d %d", &nodes_num, &edges_num, &parts_num);
    // init structures
    //int *weights = (int *) malloc(nodes_num*sizeof(int));
    int *parts = (int *) malloc(nodes_num*sizeof(int));
    int *partitions = (int *) malloc(parts_num*nodes_num*sizeof(int));
    int *mat = (int *) malloc(nodes_num*nodes_num*sizeof(int));
    // read rest of the input
    

    // setup csr representation
    int *h_csr_offsets = (int *) malloc((nodes_num + 1) * sizeof(int));
    int *h_csr_columns = (int *) malloc(edges_num * sizeof(int));
    int *h_csr_values = (int *) malloc(edges_num * sizeof(int));
    
    readInput(in_file, partitions, parts, nodes_num, edges_num, parts_num, h_csr_offsets, h_csr_columns, h_csr_values);
    //csrSetup(nodes_num, edges_num, mat, h_csr_offsets, h_csr_columns, h_csr_values);

    // setup csc representation
    int *h_csc_offsets = (int *) malloc((nodes_num + 1) * sizeof(int));
    int *h_csc_rows = (int *) malloc(edges_num * sizeof(int));
    int *h_csc_values = (int *) malloc(edges_num * sizeof(int));

    CSR *row_rep = (CSR*) malloc(sizeof(CSR));
    row_rep -> offsets = h_csr_offsets;
    row_rep -> col_indexes = h_csr_columns;
    row_rep -> values = h_csr_values;

    CSC *col_rep = (CSC*) malloc(sizeof(CSC));
    col_rep -> offsets = h_csc_offsets;
    col_rep -> row_indexes = h_csc_rows;
    col_rep -> values = h_csc_values;

    //cscSetup(nodes_num, edges_num, mat, h_csc_offsets, h_csc_rows, h_csc_values);
    cusparseSetup(row_rep, col_rep, nodes_num, edges_num);

    // generate csr rep
    // Device memory management

    csrTest(h_csr_offsets, h_csr_columns, h_csr_values, nodes_num, edges_num);
    cscTest(h_csc_offsets, h_csc_rows, h_csc_values, nodes_num, edges_num);

    

    printf("###################################\n");
    printf("#### STARTING SERIAL EXECUTION ####\n");
    printf("###################################\n");
    auto start = std::chrono::high_resolution_clock::now();
    lns_serial(partitions, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "average serial execution: " 
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MAX_ITER 
         << "us" << std::endl;

    printf("########################################\n");
    printf("#### STARTING PARALLEL_V1 EXECUTION ####\n");
    printf("########################################\n");
    start = std::chrono::high_resolution_clock::now();
    lns_v1(parts, parts_num, nodes_num, edges_num, MAX_MASS, DESTR_PERCENT, row_rep, col_rep);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "average parallel execution: " 
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MAX_ITER 
         << "us" << std::endl;
    
    free(partitions);
    //free(weights);
    free(parts);
    free(row_rep -> offsets);
    free(row_rep -> col_indexes);
    free(row_rep -> values);
    free(col_rep -> offsets);
    free(col_rep -> row_indexes);
    free(col_rep -> values);
    free(row_rep);
    free(col_rep);
    return 0;
}