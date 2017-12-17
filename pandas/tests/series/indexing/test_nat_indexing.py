# coding=utf-8
# pylint: disable-msg=E1101,W0612
import numpy as np

import pandas as pd
from pandas import Series, date_range, NaT
from pandas.util.testing import assert_series_equal


class TestNatIndexing(object):

    def setup_method(self, method):
        self.series = Series(date_range('1/1/2000', periods=10))

    # ---------------------------------------------------------------------
    # NaT support

    def test_set_none_nan(self):
        self.series[3] = None
        assert self.series[3] is NaT

        self.series[3:5] = None
        assert self.series[4] is NaT

        self.series[5] = np.nan
        assert self.series[5] is NaT

        self.series[5:7] = np.nan
        assert self.series[6] is NaT

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
