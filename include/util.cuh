float computeCost(int *int_costs, int *ext_costs, int k);
void computeRandomMask(int * mask, int n, int m);
int checkMass(int *int_costs, int parts_num, int max_mass);
void computeEdgeCost(int *parts, int part_id, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_cost, int *ext_cost);
void computeAllEdgeCost(int *parts, CSR *row_rep, CSC *col_rep, int parts_num, int nodes_num, int edges_num, int *int_costs, int *ext_costs);
float computeCost(int *int_costs, int *ext_costs, int k);
void removeFromCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep);
int addToCost(int *parts, int k, int n, int node, int *int_costs, int *ext_costs, CSR *csr_rep, CSC *csc_rep);