from pandas import DataFrame

from eran_new.data_frame_utils import get_shared_indexes


def format_dataframe(ref_mat: DataFrame, mix_mat: DataFrame) -> DataFrame:
    ref_mat = ref_mat.rename(_convert_name, axis="columns")
    mix_mat = mix_mat.rename(_convert_name, axis="columns")
    ref_mat = ref_mat.groupby(level=0, axis=1).sum()
    _, ref_mat = get_shared_indexes(mix_mat.T, ref_mat.T)
    ref_mat = ref_mat.T
    difference_indexes = mix_mat.columns.difference(ref_mat.columns)
    ref_mat[difference_indexes] = 0.0
    return ref_mat


def _convert_name(cell_name: str) -> str:
    cell_name = cell_name.lower()
    if "endothelial" in cell_name or "endo" in cell_name:
        return "Endothelial"
    if "neutrophil" in cell_name:
        return "Neutrophils"
    if "macrophage" in cell_name:
        return "Macrophages"
    if "fibroblast" in cell_name or "cafs" in cell_name:
        return "Fibroblasts"
    if "cd4" in cell_name:
        return "CD4 T Cells"
    if "cd8" in cell_name:
        return "CD8 T Cells"
    if "mono" in cell_name or "cd14" in cell_name:
        return "Monocytes"
    if "nk" in cell_name or "natural" in cell_name:
        return "NK Cells"
    if "mast" in cell_name:
        return "Mast Cell"
    if (
        cell_name == "b"
        or "cd19b" in cell_name
        or "b cell" in cell_name
        or "b_cell" in cell_name
        or "b.cell" in cell_name
        or "bcell" in cell_name
        or "b lineage" in cell_name
        or "b-cell" in cell_name
        and "pro" not in cell_name
    ):
        return "B Cells"
    if cell_name in ["p-value", "correlation", "rmse", "absolute score (sig.score)"]:
        return "trash"
    return "Other"
