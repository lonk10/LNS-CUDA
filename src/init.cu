#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>           
#include <stdlib.h>
#include "../include/lns.cuh"
#include "../include/init.cuh"


// Input read function
void readInput(FILE *in_file, int *partitions, int *weights, int *parts, int nodes_num, int edges_num, int parts_num, int *mat){

    // gather nodes (weigth and partition)
    int weight;
    char line[100];
    for (int i = 0; i < nodes_num*parts_num; i++){
        partitions[i] = 0;
    }

    //printf("Number of nodes: %d \nNumber of edges: %d\n", nodes_num, edges_num);
    //printf("Gathering nodes...\n");
    for (int i = 0; i < nodes_num; i++){
        fgets(line, 100, in_file);
        sscanf(line, "%d %d", &weights[i], &parts[i]);
        partitions[parts[i]*nodes_num+i] = 1;
        printf("Assigned node %d to partition %d\n", i, parts[i]);
    }

    //checkPartsPerNode(partitions, parts_num, nodes_num);
    // gather edges
    printf("Gathering edges...\n");
    for (int i = 0; i < nodes_num*nodes_num; i++){
        mat[i] = 0;
    }
    int n1, n2;
    for (int i = 0; i < edges_num; i++){
        fgets(line, 100, in_file);
        sscanf(line, "%d %d %d", &n1, &n2, &weight);
        mat[n1 * nodes_num + n2] = weight;
    }
    fclose(in_file);
    printf("Input read.\n");
}


// Given a sparse matrix nodes_num*nodes_num with edges_num non-zero-elements, puts 
// a csr representation into offsets, columns, values

void csrSetup(int nodes_num, int edges_num, int *mat, int *h_csr_offsets, int *h_csr_columns, int *h_csr_values){
    printf("CSR setup start...\n");
    int *d_csr_offsets, *d_csr_columns;
    int *d_csr_values,  *d_dense;
    int dense_size = nodes_num * nodes_num;

    CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (nodes_num + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMemcpy(d_dense, mat, dense_size * sizeof(int),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, nodes_num, nodes_num, nodes_num, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, nodes_num, nodes_num, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    printf("Conversion...\n");
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(int)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // device result check
    CHECK_CUDA( cudaMemcpy(h_csr_offsets, d_csr_offsets,
        (nodes_num + 1) * sizeof(int),
        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csr_columns, d_csr_columns, edges_num * sizeof(int),
            cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csr_values, d_csr_values, edges_num * sizeof(int),
            cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA( cudaFree(d_dense) )
}


// Given a sparse matrix nodes_num*nodes_num with edges_num non-zero-elements, puts 
// a csc representation into offsets, columns, values

void cscSetup(int nodes_num, int edges_num, int *mat, int *h_csc_offsets, int *h_csc_columns, int *h_csc_values){
    printf("CSC setup start...\n");
    int *d_csc_offsets, *d_csc_columns;
    int *d_csc_values,  *d_dense;
    int dense_size = nodes_num * nodes_num;

    CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &d_csc_offsets,
                           (nodes_num + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMemcpy(d_dense, mat, dense_size * sizeof(int),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, nodes_num, nodes_num, nodes_num, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsc(&matB, nodes_num, nodes_num, 0,
                                      d_csc_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    printf("Conversion...\n");
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csc_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csc_values,  nnz * sizeof(int)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCscSetPointers(matB, d_csc_offsets, d_csc_columns,
                                           d_csc_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // device result check
    CHECK_CUDA( cudaMemcpy(h_csc_offsets, d_csc_offsets,
        (nodes_num + 1) * sizeof(int),
        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csc_columns, d_csc_columns, edges_num * sizeof(int),
            cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csc_values, d_csc_values, edges_num * sizeof(int),
            cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csc_offsets) )
    CHECK_CUDA( cudaFree(d_csc_columns) )
    CHECK_CUDA( cudaFree(d_csc_values) )
    CHECK_CUDA( cudaFree(d_dense) )
}

// test for correct csr build

void csrTest(int *offsets, int *columns, int *values, int n, int e){
    // csr test results
    int   h_csr_offsets_result[]  = { 0, 2, 5, 7, 9, 9, 9, 11, 11, 13, 14 };
    int   h_csr_columns_result[]  = { 4, 5, 4, 5, 6, 3, 4, 7, 8, 7, 9, 7, 9, 7 };
    int h_csr_values_result[]   = { 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1};
    int correct = 1;
    printf("Testing...\n");

    for (int i = 0; i < n + 1; i++) {
        if (offsets[i] != h_csr_offsets_result[i]) {
            correct = 0;break;
        }
    }
    if (correct)
        printf("offset test PASSED\n");
    else
        printf("offset test FAILED: wrong result\n");
    correct = 1;

    for (int i = 0; i < e; i++) {
        if (columns[i] != h_csr_columns_result[i]) {
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
        if (values[i] != h_csr_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("values test PASSED\n");
    else
        printf("values test FAILED: wrong result\n");

}

// test for correct csc build

void cscTest(int *offsets, int *rows, int *values, int n, int e){
    // csr test results
    int   h_csc_offsets_result[]  = { 0, 0, 0, 0, 1, 4, 6, 7, 11, 12, 14 };
    int   h_csc_rows_result[]   = { 2, 0, 1, 2, 0, 1, 1, 3, 6, 8, 9, 3, 6, 8 };
    int h_csc_values_result[]   = { 2, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1 };
    int correct = 1;
    printf("Testing...\n");

    for (int i = 0; i < n + 1; i++) {
        if (offsets[i] != h_csc_offsets_result[i]) {
            printf("offset %d should be %d and is %d\n", i, h_csc_offsets_result[i], offsets[i]);
            correct = 0;break;
        }
    }
    if (correct)
        printf("offset test PASSED\n");
    else
        printf("offset test FAILED: wrong result\n");
    correct = 1;

    for (int i = 0; i < e; i++) {
        if (rows[i] != h_csc_rows_result[i]) {
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
        if (values[i] != h_csc_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("values test PASSED\n");
    else
        printf("values test FAILED: wrong result\n");

}