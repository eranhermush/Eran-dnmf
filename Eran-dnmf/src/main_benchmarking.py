from os import mkdir, listdir
from os.path import isdir

from benchmarking import run_nmf_on_data
from utils_functions import load_nmf

sig_folder_pkl = "/content/drive/MyDrive/TAU/Roded/Nmf-Objects-2"
datasets_folder = "/content/drive/MyDrive/TAU/Roded/Mixes"
dataref_folder = "/content/drive/MyDrive/TAU/Roded/refMat"
output_folder = "/content/drive/MyDrive/TAU/Roded/Nmf-results"

output_folder = "../resources/gedit/NMF-obj"

signames = [f for f in listdir(sig_folder_pkl)]

for sign_name in signames:
    print(sign_name)
    sig_name_only = sign_name.split("_")[0]
    ref_path = f"{dataref_folder}/{sig_name_only}.tsv"
    signature_dir = f"{output_folder}/{sign_name}"
    if not isdir(signature_dir):
        mkdir(signature_dir)
    sign_options_names = [g for g in listdir(f"{sig_folder_pkl}/{sign_name}")]
    for sign_option in sign_options_names:
        print(f"sign option: {sign_option}")
        signature_path = f"{sig_folder_pkl}/{sign_name}/{sign_option}"
        model = load_nmf(signature_path)
        mixes_names = [h for h in listdir(datasets_folder)]
        for mix_name in mixes_names:
            print(f"Mix name : {mix_name}")
            mix_signature_folder = f"{signature_dir}/{mix_name}"
            if not isdir(mix_signature_folder):
                mkdir(mix_signature_folder)
            output_path = f"{mix_signature_folder}/{sign_option}"
            mix_path = f"{datasets_folder}/{mix_name}"
            run_nmf_on_data(mix_path, ref_path, output_path, model["model"])
