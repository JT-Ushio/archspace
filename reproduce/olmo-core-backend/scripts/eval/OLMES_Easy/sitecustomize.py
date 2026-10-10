"""Process-wide runtime fixes for OLMES evaluation jobs.

OLMES starts one model process per data-parallel worker.  Python's Linux
default is ``fork``, but a forked child cannot initialize CUDA after the
parent has queried CUDA.  ``sitecustomize`` is imported automatically by
Python, so setting ``spawn`` here also applies to the subprocess started by
``oe_eval.launch`` without modifying the upstream OLMES checkout.
"""

import argparse as _argparse
import multiprocessing as _multiprocessing


_multiprocessing.set_start_method("spawn", force=True)


# OLMES declares ``--save-raw-requests`` with ``type=bool``.  Argparse then
# evaluates the generated child argument ``false`` as ``bool("false")``, which
# is True and can write several GB of request JSONL during an easy-suite run.
# Keep the upstream checkout untouched and correct only this one option.
_original_get_value = _argparse.ArgumentParser._get_value


def _get_value(self, action, arg_string):
    if action.dest == "save_raw_requests" and action.type is bool:
        normalized = str(arg_string).strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return _original_get_value(self, action, arg_string)


_argparse.ArgumentParser._get_value = _get_value
