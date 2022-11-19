#!/bin/bash


for((i=0; i<1; i++)); do {
	      python main.py --sample --config ffhq.yml  --model_dir ffhq  -i ./cs_results
        echo "DONE!"
} & done
wait

