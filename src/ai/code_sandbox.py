"""Sandboxed code interpreter for arbitrary xarray / pandas / matplotlib scripts.

The user-facing tool is `execute_xarray_script`.  It validates the submitted
script, runs it in a fresh OS process with a restricted namespace, and returns
the captured stdout plus any figure paths saved to `figures/`.

The sandbox is deliberately defensive:

* No `import` / `from ... import` statements in user code.
* No raw `open`, `eval`, `exec`, `__import__` etc. in the restricted builtins.
* Only a whitelist of built-ins is exposed.
* File I/O on `xarray`, `pandas` and `numpy` is patched or blocked; the script
  must use `load_dataset()` / `save_figure()` / `plt.savefig()` / `fig.savefig()`.
* The child process is started by `subprocess.run` with a timeout, so an
  infinite loop or heavy computation is killed.

This is a *practical* sandbox, not a formally verified one. If you need
military-grade isolation, replace the subprocess with a container.
"""

from __future__ import annotations

import ast
import base64
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from io import StringIO
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "demo"
LARGE_DATA_DIR = BASE_DIR / "largeData"
FIGURES_DIR = BASE_DIR / "figures"

# Extra data directories the sandbox is allowed to read from
ALLOWED_DATA_DIRS = [d for d in [DATA_DIR, LARGE_DATA_DIR] if d.exists()]


# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------

_FORBIDDEN_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
    "breakpoint",
    "input",
    "quit",
    "exit",
    "copyright",
    "credits",
    "license",
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "globals",
    "vars",
    "locals",
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "urllib",
    "requests",
    "importlib",
    "tempfile",
    "glob",
    "fnmatch",
    "multiprocessing",
    "concurrent",
    "threading",
    "asyncio",
    "webbrowser",
    "http",
    "ftplib",
    "smtplib",
    "email",
}

_FORBIDDEN_ATTRS = {
    "__class__",
    "__base__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__kwdefaults__",
    "__dict__",
    "__module__",
    "__file__",
    "__path__",
    "__loader__",
    "__cached__",
    "__package__",
    "__spec__",
    "__builtins__",
    "__import__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    # file I/O attributes that should not be called directly
    "to_netcdf",
    "to_zarr",
    "to_csv",
    "to_pickle",
    "to_hdf",
    "to_sql",
    "to_excel",
    "to_json",
    "to_html",
    "to_latex",
    "to_markdown",
    "to_clipboard",
    "save",
    "savetxt",
    "load",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "savemat",
    "loadmat",
    "open_mfdataset",
    "open_dataarray",
    "open_zarr",
    "imsave",
}


class _ScriptValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append("Import statements are not allowed in the sandbox")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append("From-import statements are not allowed in the sandbox")

    def visit_Global(self, node: ast.Global) -> None:
        self.errors.append("Global statements are not allowed in the sandbox")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.errors.append("Nonlocal statements are not allowed in the sandbox")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.errors.append("Class definitions are not allowed in the sandbox")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.errors.append("Async functions are not allowed in the sandbox")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self.errors.append(f"Forbidden name '{node.id}' used at line {node.lineno}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_ATTRS:
            self.errors.append(
                f"Forbidden attribute '{node.attr}' used at line {node.lineno}"
            )
        self.generic_visit(node)


def _validate_script(script: str) -> list[str]:
    """Return a list of security / policy violations for the user script."""
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        return [f"Syntax error in user script: {exc}"]

    validator = _ScriptValidator()
    validator.visit(tree)
    return validator.errors


# ---------------------------------------------------------------------------
# Built-in whitelist
# ---------------------------------------------------------------------------

import builtins as _builtins

_SAFE_BUILTINS = (
    # constants
    "None True False Ellipsis NotImplemented\n"
    # types / constructors
    "bool bytearray bytes complex dict float frozenset int list set str tuple\n"
    # functions
    "abs all any ascii bin callable chr classmethod divmod enumerate filter "
    "format hex id isinstance issubclass iter len map max memoryview min next "
    "object oct ord pow property range repr reversed round slice sorted "
    "staticmethod sum super type zip\n"
    # exceptions
    "BaseException Exception ArithmeticError AssertionError AttributeError "
    "BlockingIOError BrokenPipeError BufferError BytesWarning ChildProcessError "
    "ConnectionAbortedError ConnectionError ConnectionRefusedError "
    "ConnectionResetError DeprecationWarning EOFError EnvironmentError "
    "FileExistsError FileNotFoundError FloatingPointError FutureWarning "
    "GeneratorExit IOError ImportError ImportWarning IndentationError "
    "IndexError InterruptedError IsADirectoryError KeyError KeyboardInterrupt "
    "LookupError MemoryError ModuleNotFoundError NameError NotADirectoryError "
    "NotImplementedError OSError OverflowError PendingDeprecationWarning "
    "PermissionError ProcessLookupError RecursionError ReferenceError "
    "ResourceWarning RuntimeError RuntimeWarning StopAsyncIteration "
    "StopIteration SyntaxError SyntaxWarning SystemError SystemExit TabError "
    "TimeoutError TypeError UnboundLocalError UnicodeDecodeError "
    "UnicodeEncodeError UnicodeError UnicodeTranslationError UnicodeWarning "
    "UserWarning ValueError Warning ZeroDivisionError"
)

_SAFE_BUILTINS_DICT = {
    name: getattr(_builtins, name)
    for name in _SAFE_BUILTINS.split()
    if hasattr(_builtins, name)
}


# ---------------------------------------------------------------------------
# Dataset pre-loading and safe I/O helpers
# ---------------------------------------------------------------------------

def _find_demo_datasets() -> dict[str, Path]:
    """Collect available NetCDF files from data/demo and largeData."""
    mapping: dict[str, Path] = {}
    for root in ALLOWED_DATA_DIRS:
        for path in root.rglob("*.nc"):
            key = path.stem
            mapping[key] = path
    return mapping


def _is_under_data_dir(path: Path) -> bool:
    """Return True if a resolved path is inside an allowed data directory."""
    try:
        resolved = path.resolve()
    except Exception:
        return False
    for allowed in ALLOWED_DATA_DIRS:
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            pass
    return False


def _is_under_figures_dir(path: Path) -> bool:
    """Return True if a resolved path is inside the project's figures directory."""
    try:
        return path.resolve().is_relative_to(FIGURES_DIR.resolve())
    except Exception:
        return False


def _setup_sandbox_globals() -> dict:
    """Prepare the restricted namespace used by the user script."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; safe on headless servers
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import xarray as xr

    # Snapshot original I/O functions before patching
    _orig_xr_open_dataset = xr.open_dataset
    _orig_plt_savefig = plt.savefig

    from matplotlib.figure import Figure

    _orig_fig_savefig = Figure.savefig

    # Keep track of figures saved by the user code
    _saved_figures: list[str] = []

    # Pre-open available datasets (lazy by default)
    _dataset_paths = _find_demo_datasets()
    _datasets: dict[str, xr.Dataset] = {}
    for key, path in _dataset_paths.items():
        try:
            _datasets[key] = _orig_xr_open_dataset(path)
        except Exception:
            pass

    def _safe_open_dataset(path, *args, **kwargs):
        p = Path(path)
        if not p.is_absolute():
            p = (DATA_DIR / p)
        p = p.resolve()
        if not _is_under_data_dir(p):
            raise ValueError(f"Sandbox: cannot open {p}; it is outside the data directory.")
        return _orig_xr_open_dataset(p, *args, **kwargs)

    def _safe_load_dataset(name_or_path):
        if name_or_path in _datasets:
            return _datasets[name_or_path]
        return _safe_open_dataset(name_or_path)

    def _safe_figure_savefig(self, fname=None, *args, **kwargs):
        if fname is None:
            fname = f"agent_sandbox_{uuid4().hex[:8]}.png"
        p = Path(fname)
        if not p.is_absolute():
            p = FIGURES_DIR / p
        else:
            p = p.resolve()
        if not _is_under_figures_dir(p):
            raise ValueError(f"Sandbox: can only save figures inside {FIGURES_DIR}.")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        _orig_fig_savefig(self, p, *args, **kwargs)
        _saved_figures.append(str(p))
        return str(p)

    def _safe_plt_savefig(fname=None, *args, **kwargs):
        fig = plt.gcf()
        return _safe_figure_savefig(fig, fname, *args, **kwargs)

    def _safe_save_figure(fig=None, name=None):
        if fig is None:
            fig = plt.gcf()
        if name is None:
            name = f"{uuid4().hex[:8]}.png"
        if not name.endswith(".png"):
            name += ".png"
        p = FIGURES_DIR / f"agent_sandbox_{name}"
        return _safe_figure_savefig(fig, p)

    def _block(name: str):
        def _blocked(*args, **kwargs):
            raise RuntimeError(
                f"{name} is not allowed in the sandbox. "
                "Use load_dataset() / save_figure() / plt.savefig() for I/O."
            )
        return _blocked

    # Patch xarray
    xr.open_dataset = _safe_open_dataset
    xr.open_mfdataset = _block("xr.open_mfdataset")
    xr.open_dataarray = _block("xr.open_dataarray")
    xr.open_zarr = _block("xr.open_zarr")
    xr.save_mfdataset = _block("xr.save_mfdataset")
    try:
        xr.Dataset.to_netcdf = _block("Dataset.to_netcdf")
        xr.DataArray.to_netcdf = _block("DataArray.to_netcdf")
    except Exception:
        pass
    try:
        xr.Dataset.to_zarr = _block("Dataset.to_zarr")
    except Exception:
        pass

    # Patch pandas file I/O
    _pd_block = _block("pandas file I/O")
    for _pd_name in [
        "read_csv", "read_table", "read_fwf", "read_excel", "read_json",
        "read_html", "read_sql", "read_sql_table", "read_sql_query",
        "read_clipboard", "read_pickle",
    ]:
        if hasattr(pd, _pd_name):
            setattr(pd, _pd_name, _pd_block)
    for _pd_method in [
        "to_csv", "to_pickle", "to_hdf", "to_sql", "to_excel", "to_json",
        "to_html", "to_latex", "to_markdown", "to_clipboard",
    ]:
        try:
            setattr(pd.DataFrame, _pd_method, _block(f"DataFrame.{_pd_method}"))
            setattr(pd.Series, _pd_method, _block(f"Series.{_pd_method}"))
        except Exception:
            pass

    # Patch numpy file I/O
    for _np_name in ["save", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "savez", "savez_compressed"]:
        if hasattr(np, _np_name):
            setattr(np, _np_name, _block(f"np.{_np_name}"))

    # Patch matplotlib
    plt.savefig = _safe_plt_savefig
    Figure.savefig = _safe_figure_savefig
    if hasattr(plt, "imsave"):
        plt.imsave = _block("plt.imsave")
    if hasattr(plt, "imread"):
        plt.imread = _block("plt.imread")

    # Patch scipy file I/O
    try:
        import scipy.io as _scipy_io
        _scipy_io.loadmat = _block("scipy.io.loadmat")
        _scipy_io.savemat = _block("scipy.io.savemat")
    except Exception:
        pass

    # Redirect print to a StringIO so the worker can capture it
    _output_buffer = StringIO()
    _real_print = _builtins.print

    def _sandbox_print(*args, **kwargs):
        kwargs.pop("file", None)
        _real_print(*args, file=_output_buffer, **kwargs)

    # Assemble the globals.  Note: the script can read `__name__`, but cannot
    # mutate the built-ins because the dict is a fresh copy.
    safe_globals = {
        "__builtins__": dict(_SAFE_BUILTINS_DICT),
        "__name__": "__sandbox__",
        "__doc__": None,
        "np": np,
        "pd": pd,
        "xr": xr,
        "plt": plt,
        "scipy": scipy,
        "DATASETS": _datasets,
        "DATASET_PATHS": {k: str(v) for k, v in _dataset_paths.items()},
        "load_dataset": _safe_load_dataset,
        "save_figure": _safe_save_figure,
        "print": _sandbox_print,
        "len": len,
        "range": range,
    }
    safe_globals["__builtins__"]["print"] = _sandbox_print

    return safe_globals, _output_buffer, _saved_figures


def _execute_user_script(script: str) -> dict:
    """Run the validated script in a restricted in-process environment."""
    errors = _validate_script(script)
    if errors:
        return {"stdout": "", "figures": [], "error": "\n".join(errors)}

    safe_globals, output_buffer, saved_figures = _setup_sandbox_globals()
    safe_locals: dict = {}

    try:
        exec(script, safe_globals, safe_locals)
    except BaseException:
        return {
            "stdout": output_buffer.getvalue(),
            "figures": saved_figures,
            "error": traceback.format_exc(),
        }

    return {
        "stdout": output_buffer.getvalue(),
        "figures": saved_figures,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Subprocess worker entry point
# ---------------------------------------------------------------------------

def _run_sandbox_worker() -> None:
    """Entry point run inside the child process. Reads script from stdin."""
    script = sys.stdin.read()
    result = _execute_user_script(script)
    payload = {
        "stdout_b64": base64.b64encode(result["stdout"].encode("utf-8")).decode("ascii"),
        "figures": result["figures"],
        "error": result["error"],
    }
    print("###SANDBOX_BEGIN###")
    print(json.dumps(payload))
    print("###SANDBOX_END###")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def execute_xarray_script(script: str, timeout: int = 60) -> str:
    """Execute a small xarray/pandas/matplotlib script in a sandbox.

    This is the agent's "code interpreter fallback" for questions that are not
    covered by the existing tools. The script must not contain any `import`
    statements; the following objects are pre-loaded:

        - `xr` (xarray), `np` (numpy), `pd` (pandas), `plt` (matplotlib.pyplot)
        - `scipy`
        - `DATASETS` (dict of pre-opened demo NetCDFs by file stem)
        - `load_dataset(name_or_path)`
        - `save_figure(fig=None, name=None)`

    Parameters
    ----------
    script
        Python source code to run.
    timeout
        Maximum wall-clock seconds before the child is killed.

    Returns
    -------
    str
        Captured stdout, any generated figure paths, and an error message if the
        script failed.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Validate syntax in the parent as well, so we fail fast on obvious errors.
    try:
        ast.parse(script)
    except SyntaxError as exc:
        return f"❌ Syntax error in script: {exc}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-c",
        "from src.ai.code_sandbox import _run_sandbox_worker; _run_sandbox_worker()",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=BASE_DIR,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"❌ Script timed out after {timeout} seconds."
    except Exception as exc:
        return f"❌ Could not start sandbox: {exc}"

    if proc.returncode != 0:
        return (
            f"❌ Sandbox process exited with code {proc.returncode}.\n"
            f"stderr: {proc.stderr}\n"
            f"stdout: {proc.stdout}"
        )

    match = re.search(
        r"###SANDBOX_BEGIN###\n(.*?)\n###SANDBOX_END###",
        proc.stdout,
        re.DOTALL,
    )
    if not match:
        return (
            f"❌ Could not parse sandbox output.\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return f"❌ Sandbox output was not valid JSON: {exc}.\nRaw: {match.group(1)}"

    stdout = base64.b64decode(payload.get("stdout_b64", "")).decode("utf-8", errors="replace")
    figures = payload.get("figures", [])
    error = payload.get("error")

    parts = []
    if stdout.strip():
        parts.append("**Sandbox output:**\n```\n" + stdout.strip() + "\n```")
    else:
        parts.append("**Sandbox output:** (no output)")

    if figures:
        parts.append("**Saved figures:**")
        for f in figures:
            parts.append(f"- {f}")
    else:
        parts.append("**Saved figures:** none")

    if error:
        parts.append(f"**Error:**\n```\n{error}\n```")

    return "\n\n".join(parts)


if __name__ == "__main__":
    # Manual test: run an example script from stdin
    _run_sandbox_worker()
