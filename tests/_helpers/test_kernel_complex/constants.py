from __future__ import annotations

import numpy as np


FLOAT = np.complex64


DOUBLE = np.complex128


FLOAT_SCALAR_FORMS = [
    ("scale_by_float", lambda a: a * 2.0),
    ("add_float", lambda a: a + 2.0),
    ("sub_float", lambda a: a - 2.0),
    ("div_by_float", lambda a: a / 2.0),
    ("pow_float", lambda a: a ** 2.0),
    ("rsub_float", lambda a: 2.0 - a),
    ("rdiv_float", lambda a: 2.0 / a),
    ("rpow_float", lambda a: 2.0 ** a),
]


SCOPE_64 = np.complex64(3 + 4j)


SCOPE_128 = np.complex128(3 + 4j)


SCOPE_PY = 3 + 4j
