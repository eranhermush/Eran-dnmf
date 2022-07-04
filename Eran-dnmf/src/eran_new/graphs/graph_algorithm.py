from pathlib import Path

from attr import define


@define
class GraphAlgorithm:
    path: Path
    use_signature: bool
    glob_signature: str
    algorithm_description: str
    use_true_prop: bool
    name: str
    save_normalize_graph: bool
