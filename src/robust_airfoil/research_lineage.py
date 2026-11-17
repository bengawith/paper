"""Explicit, non-overlapping paths for preserved scientific experiment revisions."""
from pathlib import Path


def resolve_lineage(root: Path, name: str) -> dict[str, Path]:
    """Resolve a declared revision without creating, modifying or reading files."""
    revisions = {
        'rebuilt': ('robust-v2-rebuilt-20261014', 'rebuilt_20261014', 'REBUILT_MODEL'),
        'controlled': ('robust-v2-controlled-20261020', 'controlled_20261020', 'CONTROLLED_MODEL'),
    }
    if name not in revisions:
        raise ValueError('Unknown research lineage; choose an explicitly declared revision')
    result_name, data_name, report_name = revisions[name]
    return {
        'results': root/'results/robust_v2/lineages'/result_name,
        'data': root/'data/robust_v2/processed'/data_name,
        'reports': root/'reports/robust_v2/external_academic_review_20261014'/report_name,
    }


def load_optimisation_for_lineage(root: Path, name: str):
    """Apply only the predeclared incidence refinement for the controlled study.

    The mission endpoints, lift targets/weights, physical uncertainty and
    engineering bounds are unchanged. The historical five-angle baseline is
    retained separately; its grid must not silently determine service accuracy.
    """
    from robust_airfoil.config import OptimisationConfig, load_yaml

    resolve_lineage(root, name)
    settings = load_yaml(root/'configs/robust_v2/optimisation.yaml')
    if name == 'controlled':
        protocol = load_yaml(root/'configs/robust_v2/controlled_study.yaml')
        if protocol.get('schema') != 'controlled-cst-scientific-protocol-v1':
            raise ValueError('Unknown controlled scientific protocol')
        settings['service_alpha_deg'] = protocol['service_alpha_deg']
    return OptimisationConfig.model_validate(settings)
