#!/bin/bash

declare -gAx ORIG_ENV

ORIG_ENV[$$_NCCL_INCLUDE_DIR]=$NCCL_INCLUDE_DIR
ORIG_ENV[$$_NCCL_LIB_DIR]=$NCCL_LIB_DIR
ORIG_ENV[$$_NCCL_ROOT_DIR]=$NCCL_ROOT_DIR
ORIG_ENV[$$_NCCL_SOCKET_IFNAME]=$NCCL_SOCKET_IFNAME
#ORIG_ENV[$$_LD_LIBRARY_PATH]=$LD_LIBRARY_PATH
ORIG_ENV[$$_LD_PRELOAD]=$LD_PRELOAD

export NCCL_INCLUDE_DIR=$CONDA_PREFIX/include
export NCCL_LIB_DIR=$CONDA_PREFIX/lib
export NCCL_ROOT_DIR=$CONDA_PREFIX
export NCCL_SOCKET_IFNAME=^docker0,lo
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$NCCL_ROOT_DIR/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_LIB_DIR/libnccl.so.2.8.4:$LD_PRELOAD

# This env variable is set by the conda krb5 package, but breaks ssh'ing to
# learnfair machines. Uninstalling the package is not an option as it's required
# for the conda cmake package.
unset KRB5CCNAME
