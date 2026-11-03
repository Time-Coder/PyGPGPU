"""kernel_import -- the example package the three same-named backend files live in.

Each module in this package is written once per backend, side by side:

    vecops.cl    -> opencl
    vecops.cu    -> cuda
    vecops.hip   -> hip

`from kernel_import import vecops` imports the MERGED module: one name, one set of python
objects, and every kernel in it is used exactly like a `@kernel` function (see
../15_kernel_import.py).

The directory is a real python package on purpose, so the dotted name `kernel_import.vecops`
resolves through this package's `__path__` -- a finder that only walked `sys.path` for a flat
name would miss it. Nothing is declared here; the name exists so that `import` has somewhere
to look.
"""
