from typing import Tuple

from pandas import DataFrame


def get_shared_indexes(mix_max: DataFrame, ref_mat: DataFrame) -> Tuple[DataFrame, DataFrame]:
    index_intersection = mix_max.index.intersection(ref_mat.index)
    share_mix = mix_max.loc[index_intersection]
    share_mix = share_mix.T.groupby(by=share_mix.T.columns, axis=1).sum().T
    share_ref = ref_mat.loc[index_intersection]
    return share_mix, share_ref
