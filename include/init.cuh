#ifndef INIT_H
#define INIT_H

void readInput(FILE *in_file, int *parts, int nodes_num, int edges_num, int parts_num, int *offsets, int *indexes, int *values);
void csrSetup(int nodes_num, int edges_num, int *mat, int *h_csr_offsets, int *h_csr_columns, int *h_csr_values);
void cscSetup(int nodes_num, int edges_num, int *mat, int *h_csc_offsets, int *h_csc_columns, int *h_csc_values);
void cusparseSetup(CSR *csr, CSC *csc, int n, int e);
void cscTest(int *offsets, int *rows, int *values, int n, int e);
void csrTest(int *offsets, int *rows, int *values, int n, int e);


#endif

