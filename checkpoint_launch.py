"""Optional training checkpoints and pre-construction W&B resume settings."""

from collections.abc import Mapping


_CHECKPOINT_KEYS = {
    "resume_from_checkpoint",
    "checkpoint_dir",
    "checkpoint_interval",
    "checkpoint_total_limit",
}


def parse_checkpoint_options(config):
    """Validate the optional YAML section without reading or creating files."""
    section = config.get("checkpoint", None)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError("checkpoint must be a mapping or null.")
    unknown = set(section) - _CHECKPOINT_KEYS
    if unknown:
        raise ValueError(
            "Unknown checkpoint option(s): " + ", ".join(sorted(map(str, unknown)))
        )
    options = dict(section)
    for key in ("resume_from_checkpoint", "checkpoint_dir"):
        value = options.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"checkpoint.{key} must be a nonempty path string or null."
            )
    for key in ("checkpoint_interval", "checkpoint_total_limit"):
        if key not in options:
            continue
        value = options[key]
        if key == "checkpoint_total_limit" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"checkpoint.{key} must be a positive integer.")
    return options


def prepare_wandb_config(wandb_config, *, section, checkpoint_options):
    """Reuse saved logging identity before an eager trainer constructor runs."""
    if wandb_config is None or section is None or section is False:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("wandb must be a mapping, false, or null.")
    if section.get("enabled") is False or section.get("mode") == "disabled":
        return None
    if wandb_config.get("mode") == "disabled":
        return None

    prepared = dict(wandb_config)
    for key in ("id", "resume"):
        if key in section:
            prepared[key] = section[key]
    checkpoint = checkpoint_options.get("resume_from_checkpoint")
    if checkpoint is None or any(key in prepared for key in ("id", "resume")):
        return prepared

    # Keep the optional API lazy: ordinary launches also work with older CoMLRL.
    from comlrl.utils import read_checkpoint_metadata

    identity = read_checkpoint_metadata(checkpoint).get("wandb")
    if identity:
        prepared.update(identity)
        prepared["resume"] = "allow"
    return prepared
