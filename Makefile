###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda


##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
NVCC_FLAGS= -I$(CUDA_TOOLKIT)/include -m 64 -G --compiler-options=-Wall --compiler-options=-Wextra --compiler-options=-Wpedantic --compiler-options=-Wconversion -Xcompiler "-g -pg" -g -pg
NVCC_LIBS= -lcusparse

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
OBJS = $(OBJ_DIR)/init.o $(OBJ_DIR)/util.o $(OBJ_DIR)/serial.o $(OBJ_DIR)/parallel_v1.o $(OBJ_DIR)/parallel_v0.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
#$(EXE) : $(OBJS)
#	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(NVCC_LIBS)

# Compile main .cpp file to object files:
#$(OBJ_DIR)/%.o : %.cu
#	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

serial : $(OBJS) $(OBJ_DIR)/serial.o
	$(NVCC) $(NVCC_FLAGS) $(OBJS) $(OBJ_DIR)/serial.o -o serial $(NVCC_LIBS)

main: $(OBJS) $(OBJ_DIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $(OBJS) $(OBJ_DIR)/main.o -o main $(NVCC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/init.o : $(SRC_DIR)/init.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/util.o : $(SRC_DIR)/util.cu $(INC_DIR)/lns.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/serial.o : $(SRC_DIR)/serial.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh $(INC_DIR)/util.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/parallel_v1.o : $(SRC_DIR)/parallel_v1.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh $(INC_DIR)/util.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/parallel_v0.o : $(SRC_DIR)/parallel_v0.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh $(INC_DIR)/util.cuh $(INC_DIR)/parallel_v1.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/main.o : main.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh $(INC_DIR)/util.cuh $(INC_DIR)/serial.cuh $(INC_DIR)/parallel_v1.cuh $(INC_DIR)/parallel_v0.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)