import os
import shutil

from reformat_zip import get_file_data


def update_best(src_folder, output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    options_names = [h for h in os.listdir(src_folder)]
    for option in options_names:
        option_output_folder = f"{output_folder}/{option}"
        if not os.path.isdir(option_output_folder):
            os.mkdir(option_output_folder)
        option_path = f"{src_folder}/{option}"

        mixes_names = [h for h in os.listdir(option_path)]
        for mix_name in mixes_names:

            mix_output_folder = f"{option_output_folder}/{mix_name}"

            if not os.path.isdir(mix_output_folder):
                os.mkdir(mix_output_folder)
            mix_path = f"{option_path}/{mix_name}"

            results_file = [h for h in os.listdir(mix_path)]
            results_groups = set([f"{h.split('_')[0]}_{h.split('_')[1]}" for h in results_file])
            for group in results_groups:
                files = [h for h in results_file if h.startswith(group)]
                assert len(files) == 2
                best_value = 100
                best_path_name = ""
                for file_name in files:
                    file_path = f"{mix_path}/{file_name}"
                    file_size = get_file_data(file_path)
                    if file_size < best_value:
                        best_value = file_size
                        best_path_name = file_name
                shutil.copyfile(f"{mix_path}/{best_path_name}", f"{mix_output_folder}/{best_path_name}")


if __name__ == "__main__":
    src_folder_gedit = (
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/" "Figure1/Eran/25.2_model/3-compare-new"
    )
    output_folder_gedit = (
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/" "Figure1/Eran/25.2_model/4-compare-new"
    )
    src_folder_other = (
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/"
        "Figure1/Eran/25.2_model/3-compare-new-on-other"
    )
    output_folder_other = (
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/"
        "Figure1/Eran/25.2_model/4-compare-new-on-other"
    )

    update_best(src_folder_gedit, output_folder_gedit)
    update_best(src_folder_other, output_folder_other)
