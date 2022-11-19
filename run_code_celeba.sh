#!/bin/bash

for((i=0; i<1; i++)); do {
        python main.py  --sample --config celeba.yml --model_dir celeba --i ./cs_results
	echo "DONE!"
} & done
wait


