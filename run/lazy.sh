#!/usr/bin/env bash
rm warp_debug.txt
nvcc ../src/main.cpp -o ./InstantGR -std=c++17 -x cu -O3 -arch=sm_80
./InstantGR -cap ../../stuff/visible/ariane/ariane.cap -net ../../stuff/visible/ariane/ariane.net -out out.out -npy ../../CircuitNet/routability_ir_drop_prediction/result_dir/congestion_gpdl/test_result/output_0.npy
