"""tideforge.hashing: the content sha ignores file encoding and chunking, and sees every value, dtype and shape change."""
import numpy as np

from tideforge import hashing as H


def test_content_sha():
    a = {"x": np.arange(12, dtype=np.int8).reshape(3, 4), "y": np.zeros(5, np.float32)}
    s = H.content_sha(a)
    assert s == H.content_sha(dict(reversed(list(a.items()))))                        # name order, not insertion order
    old = H.CHUNK_BYTES
    try:
        H.CHUNK_BYTES = 1                                                             # chunking never changes the hash
        assert H.content_sha(a) == s
    finally:
        H.CHUNK_BYTES = old
    assert H.content_sha(dict(a, x=a["x"].astype(np.int16))) != s                     # dtype
    assert H.content_sha(dict(a, x=a["x"].reshape(4, 3))) != s                        # shape
    b = dict(a, x=a["x"].copy())
    b["x"][2, 3] += 1
    assert H.content_sha(b) != s                                                      # one value
    assert H.content_sha({"z": np.asarray(3.0)}) != H.content_sha({"z": np.asarray(4.0)})
