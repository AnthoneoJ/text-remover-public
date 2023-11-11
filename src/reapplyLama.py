import glob
import os
import argparse
import shutil

from src.drawMask import get_base_fname

def reapply_lama(lama_output_folder, lama_input_folder, each_step_result_folder, n=0):
    lamaOutputFiles = glob.glob(os.path.join(lama_output_folder, "*_mask.*"))
    for file in lamaOutputFiles:
        #file_name = ''.join(file.split("/")[-1].split("_mask"))
        file_name = ''.join(get_base_fname(file).split("_mask"))
        shutil.copy(file, os.path.join(lama_input_folder, file_name))
        if n:
            shutil.copy(file, os.path.join(each_step_result_folder, f"{n}_{file_name}"))
        os.remove(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reapply lama')
    parser.add_argument('--lama_output_folder', type=str)
    parser.add_argument('--lama_input_folder', type=str)
    parser.add_argument('--each_step_result_folder', type=str)
    parser.add_argument('--n', type=int, default=0)
    args = parser.parse_args()

    lamaOutputFiles = glob.glob(os.path.join(args.lama_output_folder, "*_mask.*"))
    for file in lamaOutputFiles:
        file_name = ''.join(file.split("/")[-1].split("_mask"))
        shutil.copy(file, os.path.join(args.lama_input_folder, file_name))
        if args.n:
            shutil.copy(file, os.path.join(args.each_step_result_folder, f"{args.n}_{file_name}"))
        os.remove(file)