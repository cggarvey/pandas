import numpy as np
import pytest

from pandas import Series, date_range, Categorical
from pandas.tests.series.common import TestData
from pandas.util import testing as tm


class TestSeriesIndexing_Categorical(TestData):

    @pytest.fixture(scope='class')
    def abc(self):
        return Series(['a', 'b', 'c'], dtype='category')

    def test_reindex_categorical(self, abc):
        # reindexing to an invalid Categorical
        index = date_range('20000101', periods=3)
        result = abc.reindex(index)
        expected = Series(Categorical(values=[np.nan, np.nan, np.nan],
                                      categories=['a', 'b', 'c']))
        expected.index = index
        tm.assert_series_equal(result, expected)

    def test_reindex_categorical_partial(self, abc):
        # partial reindexing
        expected = Series(Categorical(values=['b', 'c'],
                                      categories=['a', 'b', 'c']))
        expected.index = [1, 2]
        result = abc.reindex([1, 2])
        tm.assert_series_equal(result, expected)

    def test_reindex_categorical_partial_nan(self, abc):
        expected = Series(Categorical(values=['c', np.nan],
                                      categories=['a', 'b', 'c']))
        expected.index = [2, 3]
        result = abc.reindex([2, 3])
        tm.assert_series_equal(result, expected)

    @pytest.fixture(scope='function')
    def bb(self):
        return Series(Categorical(["b", "b"], categories=["a", "b"]))

    def test_categorial_assigning_ops_all(self, bb):
        bb[:] = "a"
        exp = Series(Categorical(["a", "a"], categories=["a", "b"]))
        tm.assert_series_equal(bb, exp)

    def test_categorial_assigning_ops_int(self, bb):
        bb[1] = "a"
        exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
        tm.assert_series_equal(bb, exp)

    def test_categorial_assigning_ops_slice(self, bb):
        bb[bb.index > 0] = "a"
        exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
        tm.assert_series_equal(bb, exp)

    def test_categorial_assigning_ops_bools(self, bb):
        bb[[False, True]] = "a"
        exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
        tm.assert_series_equal(bb, exp)

    def test_categorial_assigning_ops_str(self, bb):
        bb.index = ["x", "y"]
        bb["y"] = "a"
        exp = Series(Categorical(["b", "a"], categories=["a", "b"]),
                     index=["x", "y"])
        tm.assert_series_equal(bb, exp)

    def test_categorial_assigning_ops_nan(self):
        # ensure that one can set something to np.nan
        s = Series(Categorical([1, 2, 3]))
        exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
        s[1] = np.nan
        tm.assert_series_equal(s, exp)