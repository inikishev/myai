import os
import pathlib
import typing as T
from collections.abc import Callable, Sequence


def get_all_files(
    dir: str,
    recursive: bool = True,
    extensions: T.Optional[str | Sequence[str]] = None,
    path_filter: T.Optional[Callable[[str], bool]] = None,
) -> list[str]:
    """ a list of full paths of all files in a given directory recursively.

    :param path: Path to search in.
    :param recursive: Whether to search recursively, defaults to True
    :param extensions: A list of strings, each file is checked against each extension using `endswith`. \
        Lowercase is automatically enforced. Defaults to None
    :param path_filter: Function that accepts a file path and returns True if that file should be added to list of all files. \
        Defaults to None
    :return: List of full paths of all files in a given directory.
    """
    all_files = []
    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None: extensions = tuple([i.lower() for i in extensions])
    if recursive:
        for root, _, files in (os.walk(dir)):
            for file in files:
                file_path = os.path.join(root, file)
                if path_filter is not None and not path_filter(file_path): continue
                if extensions is not None and not file.lower().endswith(extensions): continue
                if os.path.isfile(file_path): all_files.append(file_path)
    else:
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if path_filter is not None and not path_filter(file_path): continue
            if extensions is not None and not file.lower().endswith(extensions): continue
            if os.path.isfile(file_path): all_files.append(file_path)

    return all_files

@T.overload
def find_file_containing(dir, contains:str, recursive:bool = True, error:T.Literal[True] = True) -> str: ...
@T.overload
def find_file_containing(dir, contains:str, recursive:bool = True, error:T.Literal[False] = False) -> str | None: ...
def find_file_containing(dir, contains:str, recursive = True, error: bool = True) -> str | None:
    """Finds first file in a folder containing a certain string in its filename and returns its full path,
    or raises FileNotFoundError if none is found and `error` is True, otherwise returns None

    :param dir: Directory to search in.
    :param contains: Substring to search for in filenames.
    :param recursive: Whether to search in subdirectories recursively, defaults to True
    :param error: Whether to raise an error when no file is found, defaults to True
    :raises FileNotFoundError: If file is not found and `error` is True
    :return: Full path of the first file found containing the substring.
    """
    for f in get_all_files(dir, recursive=recursive):
        if contains in f:
            return f
    if error: raise FileNotFoundError(f"File containing {contains} not found in {dir}")
    return None # type:ignore

def listdir_fullpaths(folder) -> list[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def getfoldersizeMB(folder) -> float:
    total = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = pathlib.Path(root) / file
            total += path.stat().st_size
    return total / 1024 / 1024

__valid_fname_chars = frozenset(" -_.,()")

def is_valid_fname(string:str) -> bool:
    if len(string) == 0: return False
    return all([c in __valid_fname_chars or c.isalnum() for c in string])

def to_valid_fname(string:str, fallback = '-', empty_fallback = 'empty', maxlen = 127, valid_chars = __valid_fname_chars) -> str:
    """Makes sure filename doesn't have forbidden characters and isn't empty or too long,
    this does not ensure a valid filename as there are a lot of other rules,
    but does a fine job most of the time.

    Args:
        string (str): _description_
        fallback (str, optional): _description_. Defaults to '-'.
        empty_fallback (str, optional): _description_. Defaults to 'empty'.
        maxlen (int, optional): _description_. Defaults to 127.

    Returns:
        _type_: _description_
    """
    if len(string) == 0: return empty_fallback
    return ''.join([(c if c in valid_chars or c.isalnum() else fallback) for c in string[:maxlen]])