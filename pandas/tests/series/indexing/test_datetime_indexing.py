# coding=utf-8
# pylint: disable-msg=E1101,W0612
from datetime import datetime, timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import DatetimeIndex, Series, date_range, DataFrame
from pandas._libs import lib
from pandas.compat import range, lrange
from pandas.util import testing as tm
from pandas.util.testing import assert_series_equal, assert_frame_equal


class TestDatetimeIndexing(object):
    """
    Also test support for datetime64[ns] in Series / DataFrame
    """

    def setup_method(self, method):
        dti = DatetimeIndex(start=datetime(2005, 1, 1),
                            end=datetime(2005, 1, 10), freq='Min')
        self.series = Series(np.random.rand(len(dti)), dti)

    def test_fancy_getitem(self):
        dti = DatetimeIndex(freq='WOM-1FRI', start=datetime(2005, 1, 1),
                            end=datetime(2010, 1, 1))

        s = Series(np.arange(len(dti)), index=dti)

        assert s[48] == 48
        assert s['1/2/2009'] == 48
        assert s['2009-1-2'] == 48
        assert s[datetime(2009, 1, 2)] == 48
        assert s[lib.Timestamp(datetime(2009, 1, 2))] == 48
        pytest.raises(KeyError, s.__getitem__, '2009-1-3')

        assert_series_equal(s['3/6/2009':'2009-06-05'],
                            s[datetime(2009, 3, 6):datetime(2009, 6, 5)])

    def test_fancy_setitem(self):
        dti = DatetimeIndex(freq='WOM-1FRI', start=datetime(2005, 1, 1),
                            end=datetime(2010, 1, 1))

        s = Series(np.arange(len(dti)), index=dti)
        s[48] = -1
        assert s[48] == -1
        s['1/2/2009'] = -2
        assert s[48] == -2
        s['1/2/2009':'2009-06-05'] = -3
        assert (s[48:54] == -3).all()

    def test_dti_snap(self):
        dti = DatetimeIndex(['1/1/2002', '1/2/2002', '1/3/2002', '1/4/2002',
                             '1/5/2002', '1/6/2002', '1/7/2002'], freq='D')

        res = dti.snap(freq='W-MON')
        exp = date_range('12/31/2001', '1/7/2002', freq='w-mon')
        exp = exp.repeat([3, 4])
        assert (res == exp).all()

        res = dti.snap(freq='B')

        exp = date_range('1/1/2002', '1/7/2002', freq='b')
        exp = exp.repeat([1, 1, 1, 2, 2])
        assert (res == exp).all()

    def test_dti_reset_index_round_trip(self):
        dti = DatetimeIndex(start='1/1/2001', end='6/1/2001', freq='D')
        d1 = DataFrame({'v': np.random.rand(len(dti))}, index=dti)
        d2 = d1.reset_index()
        assert d2.dtypes[0] == np.dtype('M8[ns]')
        d3 = d2.set_index('index')
        assert_frame_equal(d1, d3, check_names=False)

        # #2329
        stamp = datetime(2012, 11, 22)
        df = DataFrame([[stamp, 12.1]], columns=['Date', 'Value'])
        df = df.set_index('Date')

        assert df.index[0] == stamp
        assert df.reset_index()['Date'][0] == stamp

    def test_series_set_value(self):
        # #1561

        dates = [datetime(2001, 1, 1), datetime(2001, 1, 2)]
        index = DatetimeIndex(dates)

        with tm.assert_produces_warning(FutureWarning,
                                        check_stacklevel=False):
            s = Series().set_value(dates[0], 1.)
        with tm.assert_produces_warning(FutureWarning,
                                        check_stacklevel=False):
            s2 = s.set_value(dates[1], np.nan)

        exp = Series([1., np.nan], index=index)

        assert_series_equal(s2, exp)

        # s = Series(index[:1], index[:1])
        # s2 = s.set_value(dates[1], index[1])
        # assert s2.values.dtype == 'M8[ns]'

    @pytest.mark.slow
    def test_slice_locs_indexerror(self):
        times = [datetime(2000, 1, 1) + timedelta(minutes=i * 10)
                 for i in range(100000)]
        s = Series(lrange(100000), times)
        s.loc[datetime(1900, 1, 1):datetime(2100, 1, 1)]

    def test_slicing_datetimes(self):

        # GH 7523

        # unique
        df = DataFrame(np.arange(4., dtype='float64'),
                       index=[datetime(2001, 1, i, 10, 00)
                              for i in [1, 2, 3, 4]])
        result = df.loc[datetime(2001, 1, 1, 10):]
        assert_frame_equal(result, df)
        result = df.loc[:datetime(2001, 1, 4, 10)]
        assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
        assert_frame_equal(result, df)

        result = df.loc[datetime(2001, 1, 1, 11):]
        expected = df.iloc[1:]
        assert_frame_equal(result, expected)
        result = df.loc['20010101 11':]
        assert_frame_equal(result, expected)

        # duplicates
        df = pd.DataFrame(np.arange(5., dtype='float64'),
                          index=[datetime(2001, 1, i, 10, 00)
                                 for i in [1, 2, 2, 3, 4]])

        result = df.loc[datetime(2001, 1, 1, 10):]
        assert_frame_equal(result, df)
        result = df.loc[:datetime(2001, 1, 4, 10)]
        assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
        assert_frame_equal(result, df)

        result = df.loc[datetime(2001, 1, 1, 11):]
        expected = df.iloc[1:]
        assert_frame_equal(result, expected)
        result = df.loc['20010101 11':]
        assert_frame_equal(result, expected)

    def test_frame_datetime64_duplicated(self):
        dates = date_range('2010-07-01', end='2010-08-05')

        tst = DataFrame({'symbol': 'AAA', 'date': dates})
        result = tst.duplicated(['date', 'symbol'])
        assert (-result).all()

        tst = DataFrame({'date': dates})
        result = tst.duplicated()
        assert (-result).all()
