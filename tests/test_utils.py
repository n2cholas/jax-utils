import numpy as np
import pytest

import jax_utils


def test_metrics():
    names = ['a', 'b', 'c']
    m = jax_utils.Metrics.from_names(*names)
    assert list(m.names()) == names
    m = m.update(a=1.0, b=2.0)
    m = m.update(a=3.0, c=4.0)
    m = m.update(a=5.0)
    expected = {'a': (1. + 3. + 5.) / 3., 'b': 2., 'c': 4.}
    for name, value in m.items():
        assert np.isclose(value, expected[name])

    m = m.reset()
    assert list(m.names()) == names
    m = m.update(a=1.0, b=1.0, c=1.0)
    for name, value in m.items():
        assert np.isclose(value, 1.0)


def test_prng_seq():
    prng_seq = jax_utils.PRNGSeq(5)
    assert prng_seq.next().shape == (2,)
    assert next(prng_seq).shape == (2,)


def test_merge_nested_dicts():
    d1 = {'a': {'b': 1, 'c': [2, 3]}, 'd': 4}
    d2 = {'a': {'e': 6}, 'f': 7}
    d3 = {}
    d4 = {'g': 8}
    expected = {'a': {'b': 1, 'c': [2, 3], 'e': 6}, 'd': 4, 'f': 7, 'g': 8}
    assert jax_utils.merge_nested_dicts(d1, d2, d3, d4) == expected


def test_partition_nested_dict():
    left_keys = {('a', 'c'), ('d',)}
    d = {'a': {'b': 1, 'c': [2, 3], 'e': 6}, 'd': 4, 'f': 7, 'g': 8}
    left, right = jax_utils.partition_nested_dict(d, left_keys)
    assert left == {'a': {'c': [2, 3]}, 'd': 4}
    assert right == {'a': {'b': 1, 'e': 6}, 'f': 7, 'g': 8}


def test_lerp():
    a = np.ones((5, 6))
    b = np.full((5, 6), 2)
    out = jax_utils.lerp(a, b, 0.9)
    np.testing.assert_allclose(out, 1.9)


def test_assert_dtype():

    @jax_utils.assert_dtype
    def f(x, y):
        return (x + y).astype(np.int32)

    x = y = np.ones(1, dtype=np.int32)
    assert f(x, y) == 2
    x = y = np.ones(1, dtype=np.float16)
    with pytest.raises(AssertionError):
        f(x, y)
