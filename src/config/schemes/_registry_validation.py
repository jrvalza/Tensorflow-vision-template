from pydantic import ValidationInfo


def validate_in_registry(
    value: str,
    info: ValidationInfo,
    path: tuple[str, ...],
    label: str,
) -> str:
    """Validate that 'value' is a registered key at ConfigRegistries.<path>."""
    if info.context is None:
        raise RuntimeError(
            "ConfigRegistries must be provided during configuration validation."
        )
    registry = info.context["registries"]
    for attr in path:
        registry = getattr(registry, attr)
    if value not in registry.keys():
        raise ValueError(
            f"Unknown {label}: {value!r}. Available: {list(registry.keys())}"
        )
    return value
