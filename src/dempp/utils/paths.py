from pathlib import Path


def check_path(path: str | Path, error_prefix: str = "Path") -> Path:
    """Check if a file exists and return a Path object.

    Args:
        path (str | Path): The file to check.
        error_prefix (str, optional): The error message prefix. Defaults to "Path".

    Returns:
        Path: The Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{error_prefix} {path} does not exist")
    return path


def check_dir(path: str | Path, error_prefix: str = "Directory") -> Path:
    """Check if a directory exists and return a Path object.

    Args:
        path (str | Path): The directory to check.
        error_prefix (str, optional): The error message prefix. Defaults to "Directory".


    Returns:
        Path: The Path object.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{error_prefix} {path} does not exist")
    return path


def make_dir(path: str | Path, parents: bool = True, exists_ok: bool = True) -> Path:
    """Create a directory if it does not exist and return a Path object.

    Args:
        path (str | Path): The directory to create.

    Returns:
        Path: The Path object.
    """

    path = Path(path)
    path.mkdir(parents=parents, exist_ok=exists_ok)
    return path
