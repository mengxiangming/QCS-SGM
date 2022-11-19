#!/bin/bash


for((i=0; i<1; i++)); do {
	python main.py --sample --config mnist.yml  --model_dir mnist  -i ./cs_results
	echo "DONE!"
} & done
wait


