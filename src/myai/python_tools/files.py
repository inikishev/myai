import os, shutil
import pathlib
import typing as T
from collections.abc import Callable, Sequence


def get_all_files(
    dir: str,
    recursive: bool = True,
    extensions: str | Sequence[str] | None = None,
    path_filter: Callable[[str], bool] | None = None,
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
    if extensions is not None: extensions = tuple(i.lower() for i in extensions)
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
    return all(c in __valid_fname_chars or c.isalnum() for c in string)

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


def flatten_dir(
    root: str,
    outdir: str,
    extensions: str | Sequence[str] | None = None,
    path_filter: Callable[[str], bool] | None = None,
    sep: str | None = ".",
    ext_override: str | None = None,
    combine_small: int | None = None,
):
    """Copy all files in a directory into another directory, flattened.

    Args:
        root (str): directory to look in.
        outdir (str): output directory
        extensions (str | Sequence[str] | None, optional): filter by extension (case insensitive). Defaults to None.
        path_filter (_type_, optional): _description_. Defaults to None.
        sep (str | None, optional): if not None, includes path to the file in the file name, with `/` replaced by sep. Defaults to '.'.
        ext_override (str | None, optional): if not None, replaces extensions of output files with this. Shouldn't include `.` in it. Defaults to None.
        combine_small (str | None, optional): integer of maximum number of files, will combines smallest total modules into one file until this is met. Defaults to None.
    """
    root = os.path.normpath(root)
    files = get_all_files(root, extensions = extensions, path_filter=path_filter)

    if combine_small is None:
        for f in files:
            if sep is not None: name = f.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
            else: name = os.path.basename(f)

            if ext_override is not None:
                if '.' in name: name = '.'.join(name.split('.')[:-1])
                name = name + f'.{ext_override}'

            shutil.copyfile(f, os.path.join(outdir, name))

        return

    # now we have to combine smallest modules
    def read(f):
        try:
            with open(f, 'r', encoding='utf8') as file: return file.read()
        except UnicodeDecodeError:
            print(f)
            with open(f, 'r', encoding='Windows-1252') as file: return file.read()

    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None:
        extensions = [i if i.startswith('.') else f'.{i}' for i in extensions]
    branches = {}

    def check_file(f):
        if extensions is None: ext = True
        else: ext = f.lower().endswith(tuple(extensions))

        if path_filter is None: pp = True
        else: pp = path_filter(f)
        return ext and pp and os.path.isfile(f)

    # dirs with files in them
    for r, dirs, fs in os.walk(root):
        full_files = [os.path.join(r, f) for f in fs]
        if any(os.path.isfile(f) and check_file(f) for f in full_files):
            branches[r] = sum(len(read(f)) for f in full_files if check_file(f))

    branches_t = sorted(branches.items(), key = lambda x: x[1])

    files_to_save: dict[str, str|None] = {f: None for f in files if check_file(f)}

    renamed = {}
    while len(files_to_save) > combine_small:
        if len(branches_t) == 0:
            break

        rr = branches_t.pop(0)
        r = rr[0]

        text = ''
        if sep is not None: name = r.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
        else: name = os.path.basename(r)

        if ext_override is not None:
            if '.' in name: name = '.'.join(name.split('.')[:-1])
            if len(name) == 0: name = 'root'
            name = name + f'.{ext_override}'

        texts = [(os.path.join(r, f), read(os.path.join(r, f))) for f in os.listdir(r) if check_file(os.path.join(r, f))]

        for f, ftext in sorted(texts, key = lambda x: len(x[1])):
            del files_to_save[f]
            if len(text) == 0: text = f'# ---- {f.replace(root, '')} -------\n\n{ftext}'
            else: text += f'\n\n# ---- {f.replace(root, '')} -------\n\n{ftext}'

            if len(files_to_save) + 1 <= combine_small: break

        files_to_save[name] = text
        renamed[name] = r


    while len(files_to_save) > combine_small:

        files_loaded = sorted([(f, t) if t is not None else (f, read(f)) for f,t in files_to_save.items()], key=lambda x: len(x[1]))
        files_lodadedd = {k:v for k,v in files_loaded}
        import difflib
        smallest, t1 = files_loaded.pop(0)
        match = difflib.get_close_matches(smallest, [i[0] for i in files_loaded], n = 1, cutoff=0)[0]
        t2 = files_lodadedd[match]

        del files_to_save[smallest],files_to_save[match]

        if smallest in renamed: smallest = renamed[smallest]
        if match in renamed: match = renamed[match]

        print(smallest)
        text = f'# ---- {smallest.replace(root, '')} -------\n\n{t1}'
        text += f'\n\n# ---- {match.replace(root, '')} -------\n\n{t2}'

        n1 = os.path.basename(smallest)
        n2 = os.path.basename(match)

        dn1 = os.path.dirname(smallest)
        dn2 = os.path.dirname(match)

        if len(dn1) < len(dn2):
            name = f'{dn1}/{n1}+{n2}'
        else:
            name = f'{dn2}/{n2}+{n1}'

        while name in files_to_save:
            import random
            name += str(random.randint(0,9))

        files_to_save[name] = text

    for f, t in files_to_save.items():
        if '/' in f or '\\' in f:
            if sep is not None: name = f.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
            else: name = os.path.basename(f)

            if ext_override is not None:
                if '.' in name: name = '.'.join(name.split('.')[:-1])
                name = name + f'.{ext_override}'
        else:
            name = f

        with open(os.path.join(outdir, name), 'w', encoding = 'utf8') as file:
            if t is None: t = read(f)
            file.write(t)