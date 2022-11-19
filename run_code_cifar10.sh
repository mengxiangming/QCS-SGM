#!/bin/bash

for((i=0; i<1; i++)); do {
	python main.py --sample --config cifar10.yml  --model_dir cifar10  -i ./cs_results
	echo "DONE!"
} & done
wait

