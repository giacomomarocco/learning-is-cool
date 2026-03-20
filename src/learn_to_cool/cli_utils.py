import sys


def parse_overrides(argv=None):
    """Parse key=value arguments from the command line.

    Usage: uv run script.py g_fb=0.5 batch_size=2048
    Returns a dict of {key: value} with values auto-cast to list/int/float/str.
    Comma-separated values (e.g. omegas=1.0,1.05,1.1) are parsed as lists of floats.
    """
    if argv is None:
        argv = sys.argv[1:]
    overrides = {}
    for arg in argv:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        if ',' in val:
            val = [float(v) for v in val.split(',')]
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        overrides[key] = val
    return overrides
