#!/bin/bash

if [ -v ORIG_ENV[@] ]; then
    export NCCL_INCLUDE_DIR=${ORIG_ENV["$$_NCCL_INCLUDE_DIR"]}
    export NCCL_LIB_DIR=${ORIG_ENV["$$_NCCL_LIB_DIR"]}
    export NCCL_ROOT_DIR=${ORIG_ENV["$$_NCCL_ROOT_DIR"]}
    export NCCL_SOCKET_IFNAME=${ORIG_ENV["$$_NCCL_SOCKET_IFNAME"]}
    export LD_LIBRARY_PATH=${ORIG_ENV["$$_LD_LIBRARY_PATH"]}
    export LD_PRELOAD=${ORIG_ENV["$$_LD_PRELOAD"]}
fi
