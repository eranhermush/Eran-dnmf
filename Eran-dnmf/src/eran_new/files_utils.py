import glob
import shutil
from pathlib import Path
from typing import List

from attr import define

from eran_new.graphs import main_graph, get_folder_graphs, create_graph


def move_pbm_to_folder(pbmc_path: Path, use_signature: bool = False, glob_signature: str = "*.tsv") -> Path:
    final_dir = pbmc_path / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)

    for pbm_folder in pbmc_path.iterdir():
        if pbm_folder.is_dir() and pbm_folder.name != final_dir.name:
            pbm_dst_folder = final_dir / pbm_folder.name
            pbm_dst_folder.mkdir(exist_ok=True)

            for pkl in glob.iglob(f"{pbm_folder}/**/{glob_signature}", recursive=True):
                pkl = Path(pkl)
                if pkl.is_file():
                    signature = pkl.parent.parent.name if use_signature else ""
                    shutil.copy(pkl, pbm_dst_folder / f"{signature},{pkl.name}")
    return final_dir


ALGO_SIZE = 3


@define
class GraphAlgorithm:
    path: Path
    use_signature: bool
    glob_signature: str
    algorithm_description: str
    use_true_prop: bool
    name: str


overfir_params = GraphAlgorithm(
    path=Path("C:\\Users\\Eran\\\Downloads\\new_results\\new_results"),
    use_signature=True,
    glob_signature="*).tsv",
    algorithm_description="Dnmf overfit",
    use_true_prop=False,
    name="dnmf overfit",
)

no_overfir_params = GraphAlgorithm(
    path=Path("C:\\Users\\Eran\\\Downloads\\2_unsupervised_6.4\\2_unsupervised_6.4\\True"),
    use_signature=False,
    glob_signature="OUT*.tsv",
    algorithm_description="Dnmf no overfit",
    use_true_prop=True,
    name="dnmf no overfit",
)
nnls_params = GraphAlgorithm(
    path=Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\nnls"
    ),
    use_signature=False,
    glob_signature="*.tsv",
    algorithm_description="nnls with Gedit",
    use_true_prop=False,
    name="nnls",
)
cibersort_params = GraphAlgorithm(
    path=Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\cibersort"
    ),
    use_signature=False,
    glob_signature="*.tsv",
    algorithm_description="Cibersort with Gedit",
    use_true_prop=False,
    name="cibersort",
)
gedit_params = GraphAlgorithm(
    path=Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\gedit"
    ),
    use_signature=False,
    glob_signature="*.tsv",
    algorithm_description="Gedit algorithm",
    use_true_prop=False,
    name="gedit algorithm",
)

def copy_and_create_graph(algo_params: GraphAlgorithm, create_new_results=False):
    new_results_path = (
        move_pbm_to_folder(algo_params.path, algo_params.use_signature, algo_params.glob_signature)
        if create_new_results
        else algo_params.path / "final_results"
    )
    for pbm_folder in new_results_path.iterdir():
        main_graph(
            pbm_folder, algo_params.name, algo_params.algorithm_description, use_true_prop=algo_params.use_true_prop
        )


def main_overfit(create_new_results=False):
    path = Path("C:\\Users\\Eran\\\Downloads\\new_results\\new_results")
    new_results_path = move_pbm_to_folder(path, True, "*).tsv") if create_new_results else path / "final_results"
    for pbm_folder in new_results_path.iterdir():
        main_graph(pbm_folder, "dnmf overfit", "Dnmf overfit", use_true_prop=False)


def main_no_overfit():
    path = Path("C:\\Users\\Eran\\\Downloads\\2_unsupervised_6.4\\2_unsupervised_6.4\\True")

    new_results_path = move_pbm_to_folder(path, False, "OUT*.tsv")
    for pbm_folder in new_results_path.iterdir():
        main_graph(pbm_folder, "dnmf no overfit", "Dnmf no overfit", use_true_prop=True)


def main_nnls():
    path = Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\nnls"
    )

    new_results_path = move_pbm_to_folder(path, False, "*.tsv")
    for pbm_folder in new_results_path.iterdir():
        main_graph(pbm_folder, "nnls", "nnls with Gedit", use_true_prop=False)


def main_cibersort():
    path = Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\cibersort"
    )

    new_results_path = move_pbm_to_folder(path, False, "*.tsv")
    for pbm_folder in new_results_path.iterdir():
        main_graph(pbm_folder, "cibersort", "Cibersort with Gedit", use_true_prop=False)


def create_best_graph(algorithms: List[GraphAlgorithm]) -> None:
    for pbm_folder in (algorithms[0].path / "final_results").iterdir():
        loss_list = []
        for algo in algorithms:
            loss_list.extend(
                get_folder_graphs(
                    algo.path / "final_results" / pbm_folder.name, algo.use_true_prop, algo.name, ALGO_SIZE
                )
            )
        sorted_lists = sorted(loss_list, key=lambda x: x[1])
        names_arr = [t[0] for t in sorted_lists]
        loss_arr = [t[1] for t in sorted_lists]
        create_graph(loss_arr, names_arr, pbm_folder.name, "Best results - all algorithms")


if __name__ == "__main__":
    nnls_params2 = GraphAlgorithm(
        path=Path(
            "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\nnls_cibersort_new\\new_data\\cibersort"
        ),
        use_signature=False,
        glob_signature="*.tsv",
        algorithm_description="cibersort with Gedit",
        use_true_prop=False,
        name="cibersort",
    )
    copy_and_create_graph(nnls_params2, True)
    #all_algo = [overfir_params, no_overfir_params, nnls_params, cibersort_params, gedit_params]
    #create_best_graph(all_algo)
