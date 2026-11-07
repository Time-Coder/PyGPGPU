"""The kernels behind tests/test_kernel_known_failures.py, one module per bug.

The generated source of one python module is ONE translation unit, so a kernel whose bug
is a compile error would take every other kernel in the module down with it. Each failing
kernel lives in its own module here (each becomes its own `.cu`), which keeps one bug's
refusal from masking the others. Import them from the test by name.
"""
