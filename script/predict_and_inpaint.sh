#!/bin/bash

mkdir -p detection_results
rm detection_results/*

mkdir -p outputs
rm outputs/*

mkdir -p results
rm results/*

mkdir -p each_step_results
rm each_step_results/*

python CRAFT-pytorch/test.py --trained_model CRAFT-pytorch/craft_mlt_25k.pth --test_folder=images --result_folder=/content/auto-text-removal/detection_results/

python src/drawMask.py --coor_input_folder detection_results --input_folder images --output_real_image True --mask_output_folder outputs

PYTHONPATH=/content/auto-text-removal/lama TORCH_HOME=$(pwd) python lama/bin/predict.py model.path=/content/auto-text-removal/lama/big-lama indir=$(pwd)/outputs outdir=$(pwd)/results

lama_n=3
if [ ! -z $1 ] 
then 
    lama_n=$1
fi

for (( i=1; i<$lama_n; i++ ))
do
    python src/drawMask.py --coor_input_folder detection_results --input_folder images --output_real_image False --radius $((i*5)) --mask_output_folder outputs
    python3 src/reapplyLama.py --lama_output_folder=results --lama_input_folder=outputs --each_step_result_folder=each_step_results --n $((i))
    PYTHONPATH=/content/auto-text-removal/lama TORCH_HOME=$(pwd) python lama/bin/predict.py model.path=/content/auto-text-removal/lama/big-lama indir=$(pwd)/outputs outdir=$(pwd)/results
done
