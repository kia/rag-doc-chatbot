import subprocess


def get_local_ollama_models() -> list[str]:
    """Returns locally available Ollama models (e.g. `llama3:8b`)."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    models: list[str] = []
    for line in lines[1:]:
        model_name = line.split()[0]
        if model_name:
            models.append(model_name)
    return models
