import numpy as np
import numpy.testing as npt

from chronoepilogi import util_mass_ts as ums



#####################################################
#                                                   #
#   Automatically generated tests, Human reviewed   #
#                                                   #
#####################################################


def test_normalize_1d_and_constant():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    z = ums.normalize(x)
    # zero mean, unit std
    npt.assert_allclose(np.mean(z), 0.0, atol=1e-12)
    npt.assert_allclose(np.std(z), 1.0, atol=1e-12)

    # constant array -> zeros (std == 0 handled)
    c = np.ones(5) * 5.0
    zc = ums.normalize(c)
    npt.assert_allclose(zc, np.zeros_like(c))


def test_rolling_window_and_moving_stats():
    a = np.array([1, 2, 3, 4])
    rw = ums.rolling_window(a, 2)
    npt.assert_array_equal(rw, np.array([[1, 2], [2, 3], [3, 4]]))

    ma = ums.moving_average(a, window=2)
    npt.assert_allclose(ma, np.array([1.5, 2.5, 3.5]))

    ms = ums.moving_std(a, window=2)
    # std([1,2]) = 0.5, std([2,3]) = 0.5, std([3,4]) = 0.5
    npt.assert_allclose(ms, np.array([0.5, 0.5, 0.5]))


def _pearson_sliding(ts_col, query):
    """Helper: compute Pearson correlation between query and every sliding subsequence of ts_col."""
    n = len(ts_col)
    m = len(query)
    out = []
    for i in range(n - m + 1):
        subseq = ts_col[i : i + m]
        # if either std is zero, treat correlation as 0 to match implementation
        if np.std(subseq) == 0 or np.std(query) == 0:
            out.append(0.0)
        else:
            r = np.corrcoef(subseq, query)[0, 1]
            out.append(r)
    return np.array(out)


def test_mass2_modified_univariate_and_constant_query():
    ts = np.arange(10).astype(float).reshape(-1, 1)  # shape (time, variables=1)
    query = np.array([0.0, 1.0, 2.0])

    corr = ums.mass2_modified(ts, query)
    # result shape should be (variables, n-m+1)
    assert corr.shape == (1, len(ts) - len(query) + 1)

    expected = _pearson_sliding(ts[:, 0], query)
    npt.assert_allclose(corr[0], expected, atol=1e-12)

    # constant query -> correlations forced to 0 (implementation handles std==0)
    const_q = np.ones_like(query)
    corr_const = ums.mass2_modified(ts, const_q)
    npt.assert_allclose(corr_const, np.zeros_like(corr_const))


def test_mass2_modified_multivariate():
    # build a 2-variable time series where each column is a shifted version
    t = np.linspace(0, 1, 12)
    col1 = np.sin(2 * np.pi * t)
    col2 = np.cos(2 * np.pi * t)
    ts = np.vstack([col1, col2]).T  # shape (time, 2)

    query = col1[2:7]  # subsequence of col1
    corr = ums.mass2_modified(ts, query)

    # correlation for first column should be ~1 for matching subsequences
    expected_col1 = _pearson_sliding(col1, query)
    expected_col2 = _pearson_sliding(col2, query)

    npt.assert_allclose(corr[0], expected_col1, atol=1e-12)
    npt.assert_allclose(corr[1], expected_col2, atol=1e-12)
    # values must be within [-1, 1]
    assert np.all(corr <= 1.0) and np.all(corr >= -1.0)



#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

