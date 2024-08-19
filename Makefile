###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda


##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
NVCC_FLAGS= -I$(CUDA_TOOLKIT)/include --compiler-options=-Wall --compiler-options=-Wextra --compiler-options=-Wpedantic --compiler-options=-Wconversion
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
OBJS = $(OBJ_DIR)/init.o

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

# Compile CUDA source files to object files:
$(OBJ_DIR)/init.o : $(SRC_DIR)/init.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/serial.o : $(SRC_DIR)/serial.cu $(INC_DIR)/init.cuh $(INC_DIR)/lns.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)