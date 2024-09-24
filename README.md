**LNS_CUDA**

Serial and CUDA implementation of a Large Neighbourhood Search for directed weighted graphs

**Compilation**

To compile simply use
`make main` 
in the main directory.
The CUDA toolkit and a CUDA-capable device are needed.



**Usage**

Once compiled, run `main /path/to/your/input/file`

The input file *MUST* be of the form:
```
number_of_nodes number_of_edges number_of_partitions
node_1 partition // list of all nodes and their respective partition
...
node_end partition
node_1 node_1 weight // list of all edges and their weight
...
```
