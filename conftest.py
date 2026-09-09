"""Present so that pytest puts the repository root on ``sys.path``.

The sweep modules (``sweep``, ``runs``, ``utils``) are imported as top-level
names, which only works when the root is importable.
"""
