#!/usr/bin/env bash
export ORT_DISABLE_TENSORRT=1
export ORT_TENSORRT_UNAVAILABLE_WARNINGS=1
export ORT_PROVIDERS=CUDAExecutionProvider
export LD_LIBRARY_PATH=""

python3 llm/src/main.py "$@"
