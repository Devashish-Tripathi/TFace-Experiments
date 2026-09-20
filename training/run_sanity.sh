#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export OMP_NUM_THREADS=1

torchrun --nproc_per_node=1 train.py --yaml_name train_sanity.yaml