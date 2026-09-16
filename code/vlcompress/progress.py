"""progress bars"""
try:
    from tqdm.auto import tqdm as _tqdm

    def bar(it, desc="", total=None, leave=False):
        return _tqdm(it, desc=desc, total=total, leave=leave, dynamic_ncols=True)
except ImportError:  # pragma: no cover
    def bar(it, desc="", total=None, leave=False):
        return it
