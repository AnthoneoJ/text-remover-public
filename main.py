VERSION = '1.1'
"""
- Added input subfolders handling

An application to automatically remove texts from images.
Forked from https://github.com/liawifelix/auto-text-removal/
Uses https://github.com/clovaai/CRAFT-pytorch for text region masking
Uses https://github.com/advimman/lama for Inpainting

Usage:
1. (Recommended) Create a new python environment.
2. Install required modules (pip install requirements.txt)
OR
pip install pyyaml easydict joblib albumentations[imgaug] hydra-core pytorch-lightning tabulate kornia webdataset \
    packaging scikit-learn scikit-image scipy torch torchvision torchaudio gdown numpy matplotlib pandas omegaconf \
    opencv-python opencv-python-headless tensorflow tensorflow-intel tqdm braceexpand easydict imageio

AND
pip install "git+https://github.com/facebookresearch/detectron2.git"
3. Set input folder path to INPUT_DIR
4. Set desired number of iterations, ITER
5. Run
"""
import os, sys, shutil, requests, zipfile, re

from tqdm import tqdm
import gdown

base_dir = os.getcwd()
craft_dir = os.path.abspath('craft_pytorch')
lama_dir = os.path.abspath('lama')
sys.path.append(craft_dir)
sys.path.append(lama_dir)

os.chdir(craft_dir)
from craft_pytorch.test_module import InferCRAFT
os.chdir(base_dir)
from src.drawMask import run_draw_mask
os.chdir(lama_dir)
from lama.bin.predict_module import run_lama
os.chdir(base_dir)
from src.reapplyLama import reapply_lama

os.chdir(base_dir)
CRAFT_MODEL_PATH = os.path.abspath('craft_pytorch/craft_mlt_25k.pth')
LAMA_CKPT_PATH = os.path.abspath('lama/big-lama/models/best.ckpt')
BIG_LAMA_DIR = os.path.abspath('lama/big-lama')
INPUT_DIR = os.path.abspath('images')
ITER = 3

# regex pattern of image filenames to include/exclude
INCLUDE_PATTERN = None
EXCLUDE_PATTERN = r'original'

input_path = os.path.abspath('input_images')
detection_path = os.path.abspath('detection_results')
outputs_path = os.path.abspath('outputs')
results_path = os.path.abspath('results')
each_step_path = os.path.abspath('each_step_results')

def download_with_progress(url: str, dest: str):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def reset_dirs():
    if not os.path.isdir(input_path):
        os.mkdir(input_path)
    else:
        shutil.rmtree(input_path)
        os.mkdir(input_path)
    if not os.path.isdir(detection_path):
        os.mkdir(detection_path)
    else:
        shutil.rmtree(detection_path) # delete entire folder
        os.mkdir(detection_path)
    if not os.path.isdir(outputs_path):
        os.mkdir(outputs_path)
    else:
        shutil.rmtree(outputs_path)
        os.mkdir(outputs_path)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    else:
        shutil.rmtree(results_path)
        os.mkdir(results_path)
    if not os.path.isdir(each_step_path):
        os.mkdir(each_step_path)
    else:
        shutil.rmtree(each_step_path)
        os.mkdir(each_step_path)

def inject_images(source_dir: str, dest_dir: str):
    image_filenames = [filename for filename in os.listdir(source_dir)
                       if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if INCLUDE_PATTERN:
        image_filenames = [filename for filename in image_filenames
                           if re.search(INCLUDE_PATTERN, filename, re.IGNORECASE)]
    if EXCLUDE_PATTERN:
        image_filenames = [filename for filename in image_filenames
                           if re.search(EXCLUDE_PATTERN, filename, re.IGNORECASE) is None]
    for filename in image_filenames:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(dest_dir, filename)
        shutil.copy2(source_path, destination_path)

def save_output(source_dir: str, dest_dir: str):
    image_filenames = [filename for filename in os.listdir(source_dir)
                       if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    output_filenames = []
    for filename in image_filenames:
        fname = filename.split('.', 1)
        ext = fname[-1]
        output_filenames.append(fname[0]+'_output'+'.'+ext)
    for source_filename,dest_filename in zip(image_filenames,output_filenames):
        source_path = os.path.join(source_dir, source_filename)
        destination_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_path, destination_path)

if __name__=='__main__':
    craft_model_url = 'https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ'
    if not os.path.exists(CRAFT_MODEL_PATH):
        gdown.download(craft_model_url, CRAFT_MODEL_PATH, quiet=False, fuzzy=True)

    lama_model_path = 'lama/big-lama.zip'
    lama_model_url = 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip?download=true'
    if not os.path.exists(LAMA_CKPT_PATH):
        if not os.path.exists(lama_model_path):
            print('Downloading big-lama.zip...')
            download_with_progress(lama_model_url, lama_model_path)
            print(f'big-lama.zip downloaded to {lama_model_path}')
        with zipfile.ZipFile(lama_model_path, 'r') as zip_ref:
            zip_ref.extractall('lama/')
            print(f'lama checkpoint extracted to {LAMA_CKPT_PATH}')

    subfolders = [os.path.join(INPUT_DIR, folder) for folder in os.listdir(INPUT_DIR)
                  if os.path.isdir(os.path.join(INPUT_DIR, folder))]
    if len(subfolders) == 0:
        subfolders = [INPUT_DIR]

    for input_folder in subfolders:
        reset_dirs()
        inject_images(input_folder, input_path)

        os.chdir(craft_dir)
        infer_craft = InferCRAFT(trained_model=CRAFT_MODEL_PATH, test_folder=input_path, result_folder=detection_path, cuda=False)
        infer_craft.run()

        os.chdir(base_dir)
        run_draw_mask(detection_path, input_path, outputs_path, output_real_image=True)

        os.chdir(lama_dir)
        run_lama(BIG_LAMA_DIR, outputs_path, results_path)

        for i in range(1, ITER):
            os.chdir(base_dir)
            run_draw_mask(detection_path, input_path, outputs_path, (i+1)*5, False)
            reapply_lama(results_path, outputs_path, each_step_path, i+1)
            os.chdir(lama_dir)
            run_lama(BIG_LAMA_DIR, outputs_path, results_path)

        save_output(results_path, input_folder)
        print('Output generated:',results_path)

    print('END')