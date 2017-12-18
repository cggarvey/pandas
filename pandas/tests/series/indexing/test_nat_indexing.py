# coding=utf-8
# pylint: disable-msg=E1101,W0612

import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from pandas import Series, date_range, NaT, DatetimeIndex, Timestamp
from pandas._libs import tslib
from pandas.util import testing as tm
from pandas.util.testing import assert_series_equal


class TestNatIndexing(object):

    @pytest.fixture(scope="function")
    def series(self):
        return Series(date_range('1/1/2000', periods=10))

    def test_set_none_nan(self, series):
        series[3] = None
        assert series[3] is NaT

        series[3:5] = None
        assert series[4] is NaT

        series[5] = np.nan
        assert series[5] is NaT

        series[5:7] = np.nan
        assert series[6] is NaT

    def test_nat_operations(self):
        # GH 8617
        s = Series([0, pd.NaT], dtype='m8[ns]')
        exp = s[0]
        assert s.median() == exp
        assert s.min() == exp
        assert s.max() == exp

    def test_round_nat(self):
        # GH14940
        s = Series([pd.NaT])
        expected = Series(pd.NaT)
        for method in ["round", "floor", "ceil"]:
            round_method = getattr(s.dt, method)
            for freq in ["s", "5s", "min", "5min", "h", "5h"]:
                assert_series_equal(round_method(freq), expected)

    def test_unique_nat(self):
        # create a list of 20 timestamps and a NaT.
        arr = [Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t)
               for t in range(20)] + [pd.NaT]
        # three copies of arr end-to-end. 63 items total in idx.
        idx = DatetimeIndex(arr * 3)
        tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
        assert idx.nunique() == 20
        assert idx.nunique(dropna=False) == 21

    def test_unique_inat(self):
        # NaT, note this is excluded
        arr = [1370745748 + t for t in range(20)] + [tslib.iNaT]
        idx = DatetimeIndex(arr * 3)
        tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
        assert idx.nunique() == 20
        assert idx.nunique(dropna=False) == 21
