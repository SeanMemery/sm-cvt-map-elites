class EvolutionError(Exception):
    pass


class ConfigurationError(EvolutionError):
    pass


class LLMError(EvolutionError):
    pass


class ParsingError(EvolutionError):
    pass


class ValidationError(EvolutionError):
    pass


class DescriptorError(EvolutionError):
    pass


class FitnessError(EvolutionError):
    pass


class StatsValidationError(EvolutionError):
    pass


def validate_stats(stats: object) -> dict:
    """Validate a stats dict returned alongside a fitness score.

    Expected format:
        {
            "stat_name": {
                "value": <any JSON-serialisable value>,
                "explanation": "<non-empty string>"
            },
            ...
        }

    Raises StatsValidationError with a descriptive message on any violation.
    Returns the validated dict.
    """
    if not isinstance(stats, dict):
        raise StatsValidationError(
            f"Stats must be a dict, got {type(stats).__name__}. "
            "Return (score, stats_dict) from your fitness function."
        )
    for key, entry in stats.items():
        if not isinstance(key, str) or not key:
            raise StatsValidationError(
                f"Stats keys must be non-empty strings, got {key!r}."
            )
        if not isinstance(entry, dict):
            raise StatsValidationError(
                f"Stats[{key!r}] must be a dict with 'value' and 'explanation', "
                f"got {type(entry).__name__}."
            )
        if "value" not in entry:
            raise StatsValidationError(
                f"Stats[{key!r}] is missing required key 'value'."
            )
        if "explanation" not in entry:
            raise StatsValidationError(
                f"Stats[{key!r}] is missing required key 'explanation'."
            )
        if not isinstance(entry["explanation"], str) or not entry["explanation"].strip():
            raise StatsValidationError(
                f"Stats[{key!r}]['explanation'] must be a non-empty string."
            )
    return stats


class StorageError(EvolutionError):
    pass


class CheckpointError(EvolutionError):
    pass
