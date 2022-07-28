import glob
import shutil
from pathlib import Path
from typing import List

from eran_new.graphs.config import overfir_params, no_overfir_params, nnls_params, cibersort_params, gedit_params
from eran_new.graphs.graph_algorithm import GraphAlgorithm
from eran_new.graphs.graphs import get_folder_graphs, create_graph, main_graph


def move_pbm_to_folder(pbmc_path: Path, use_signature: bool = False, glob_signature: str = "*.tsv") -> Path:
    final_dir = pbmc_path / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)

    for pbm_folder in pbmc_path.iterdir():
        if pbm_folder.is_dir() and pbm_folder.name != final_dir.name and pbm_folder.name != "normalized_graphs":
            pbm_dst_folder = final_dir / pbm_folder.name
            pbm_dst_folder.mkdir(exist_ok=True)

            for pkl in glob.iglob(f"{pbm_folder}/**/{glob_signature}", recursive=True):
                pkl = Path(pkl)
                if pkl.is_file():
                    signature = pkl.parent.parent.name if use_signature else ""
                    shutil.copy(pkl, pbm_dst_folder / f"{signature},{pkl.name}")
    return final_dir


def copy_and_create_graph(algo_params: GraphAlgorithm, create_new_results=False, create_graphs=True):
    new_results_path = (
        move_pbm_to_folder(algo_params.path, algo_params.use_signature, algo_params.glob_signature)
        if create_new_results
        else algo_params.path / "final_results"
    )
    if not create_graphs:
        return
    for pbm_folder in new_results_path.iterdir():
        main_graph(
            pbm_folder,
            algo_params.name,
            algo_params.algorithm_description,
            use_true_prop=algo_params.use_true_prop,
            save_normalize_graph=algo_params.save_normalize_graph,
        )


def create_best_graph(algorithms: List[GraphAlgorithm], algo_size: int = 3) -> None:
    for pbm_folder in (algorithms[0].path / "final_results").iterdir():
        loss_list = []
        for algo in algorithms:
            loss_list.extend(
                get_folder_graphs(
                    algo.path / "final_results" / pbm_folder.name, algo.use_true_prop, algo.name, algo_size
                )
            )
        sorted_lists = sorted(loss_list, key=lambda x: x[1])
        names_arr = [t[0] for t in sorted_lists]
        loss_arr = [float(t[1]) for t in sorted_lists]
        create_graph(loss_arr, names_arr, pbm_folder.name, "Best results - all algorithms")


def create_both_pbmc_graph(algorithms: List[GraphAlgorithm], algo_size: int = 3) -> None:
    result_all = []
    for algo in algorithms:
        result = {}
        for pbm_folder in (algorithms[0].path / "final_results").iterdir():
            if "PBMC" in pbm_folder.name:
                sorted_lists = get_folder_graphs(
                    algo.path / "final_results" / pbm_folder.name, algo.use_true_prop, algo.name, 100
                )
                sorted_lists = dict(sorted_lists)
                sorted_lists = {
                    x.lower().replace("pbmc1", "").replace("pbmc2", "").replace("normmix", ""): sorted_lists[x]
                    for x in set(sorted_lists)
                }

                if result == {}:
                    result = sorted_lists
                else:
                    result = {k: 0.5 * (result.get(k) + sorted_lists.get(k)) for k in set(result) & set(sorted_lists)}

        result_all.extend(sorted(result.items(), key=lambda x: x[1])[:3])

    result_all = sorted(result_all, key=lambda x: x[1])
    names_arr = [t[0] for t in result_all]
    loss_arr = [float(t[1]) for t in result_all]
    create_graph(loss_arr, names_arr, "pbmc-both", "Best results - all algorithms")


if __name__ == "__main__":
    # nnls_params2 = GraphAlgorithm(
    #     path=Path(
    #         "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\new_data\\nnls"
    #     ),
    #     use_signature=False,
    #     glob_signature="*.tsv",
    #     algorithm_description="cibersort with Gedit",
    #     use_true_prop=False,
    #     name="cibersort",
    #     save_normalize_graph=True,
    # )
    # copy_and_create_graph(nnls_params2, False)
    print("Start")
    all_algo = [overfir_params, no_overfir_params, nnls_params, cibersort_params, gedit_params]
    # copy_and_create_graph(overfir_params, True, True)
    [copy_and_create_graph(algo, True, False) for algo in all_algo]

    create_both_pbmc_graph(all_algo)
    create_best_graph(all_algo)
