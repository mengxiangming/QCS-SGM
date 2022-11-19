#!/bin/bash

for((i=0; i<1; i++)); do {
	python main.py --sample --config cifar10_neumann_net_compare.yml  --model_dir cifar10  -i ./nn_compare_results
	echo "DONE!"
} & done
wait

