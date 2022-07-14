from pathlib import Path

from eran_new.graphs.graph_algorithm import GraphAlgorithm

overfir_params = GraphAlgorithm(
    path=Path("C:\\Users\\Eran\\\Downloads\\new_results-3\\new_results"),
    use_signature=True,
    glob_signature="*).tsv",
    algorithm_description="Dnmf overfit",
    use_true_prop=False,
    name="dnmf overfit new",
    save_normalize_graph=True,
)

no_overfir_params = GraphAlgorithm(
    path=Path("C:\\Users\\Eran\\\Downloads\\2_unsupervised_6.4\\2_unsupervised_6.4\\True"),
    use_signature=False,
    glob_signature="OUT*.tsv",
    algorithm_description="Dnmf no overfit",
    use_true_prop=True,
    name="dnmf no overfit",
    save_normalize_graph=True,
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
    save_normalize_graph=True,
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
    save_normalize_graph=True,
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
    save_normalize_graph=True,
)
