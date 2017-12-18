# coding=utf-8
# pylint: disable-msg=E1101,W0612
from datetime import datetime, timedelta
from pytz import timezone as tz
import numpy as np
import pytest
from dateutil.tz import tzutc

import pandas as pd
from pandas import (DataFrame, Series, DatetimeIndex, Timestamp, date_range,
                    period_range)
from pandas._libs import lib, tslib, index as _index
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import range, lrange
from pandas.util import testing as tm
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_almost_equal, assert_index_equal)


def _tz(x):
    # handle special case for utc in dateutil
    return tzutc() if x == 'UTC' else gettz(x)


class TestDatetimeIndexing(object):
    """
    Also test support for datetime64[ns] in Series / DataFrame
    """

    def test_dti_snap(self):
        dti = DatetimeIndex(['1/1/2002', '1/2/2002', '1/3/2002', '1/4/2002',
                             '1/5/2002', '1/6/2002', '1/7/2002'], freq='D')

        res = dti.snap(freq='W-MON')
        exp = pd.date_range('12/31/2001', '1/7/2002', freq='w-mon')
        exp = exp.repeat([3, 4])
        assert_index_equal(res, exp)

        res = dti.snap(freq='B')
        exp = pd.date_range('1/1/2002', '1/7/2002', freq='b')
        exp = exp.repeat([1, 1, 1, 2, 2])
        assert_index_equal(res, exp)

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

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            s = Series().set_value(dates[0], 1.)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            s2 = s.set_value(dates[1], np.nan)

        exp = Series([1., np.nan], index=index)

        assert_series_equal(s2, exp)

        # s = Series(index[:1], index[:1])
        # s2 = s.set_value(dates[1], index[1])
        # assert s2.values.dtype == 'M8[ns]'

    @pytest.mark.slow
    def test_slice_locs_no_indexerror(self):
        times = [datetime(2000, 1, 1) + timedelta(minutes=i * 10)
                 for i in range(100000)]

        # "times" produces a DatetimeIndex ranging from
        # January 1, 2000 - November 25, 2001
        s = Series(lrange(100000), times)

        # test that IndexError is not raised when selection
        # is outside the range of the DatetimeIndex.
        selected = s.loc[datetime(1900, 1, 1):datetime(2100, 1, 1)]
        assert len(selected) == 100000

    def test_slicing_datetimes(self):
        # GH 7523
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

    def test_slicing_datetimes_with_duplicates(self):
        # duplicates
        df = DataFrame(np.arange(5., dtype='float64'),
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

    def test_frame_datetime64_duplicated_no_dupes(self):
        dates = pd.date_range('2010-07-01', end='2010-08-05')

        tst = DataFrame({'symbol': 'AAA', 'date': dates})
        dupes = tst.duplicated(['date', 'symbol'])
        assert not any(dupes)

        tst = DataFrame({'date': dates})
        dupes = tst.duplicated()
        assert not any(dupes)

    def test_frame_datetime64_duplicated_dupes(self):
        dates = pd.date_range('2010-07-01', end='2010-08-05')

        tst = DataFrame({'symbol': 'AAA', 'date': dates})
        doubled = pd.concat([tst, tst], axis=0, ignore_index=True)

        dupes = doubled.duplicated(['date', 'symbol'])
        assert sum(dupes) == len(tst)

        tst = DataFrame({'date': dates})
        dupes = doubled.duplicated()
        assert sum(dupes) == len(tst)

    def test_datetime_indexing(self):

        index = date_range('1/1/2000', '1/7/2000')
        index = index.repeat(3)

        s = Series(len(index), index=index)
        stamp = Timestamp('1/8/2000')

        pytest.raises(KeyError, s.__getitem__, stamp)
        s[stamp] = 0
        assert s[stamp] == 0

        # not monotonic
        s = Series(len(index), index=index)
        s = s[::-1]

        pytest.raises(KeyError, s.__getitem__, stamp)
        s[stamp] = 0
        assert s[stamp] == 0


class TestDatetimeIndexing_SetGet(object):

    @pytest.fixture(scope='function')
    def tz_ts(self):
        n = 50
        # testing with timezone, GH #2785
        rng = date_range('1/1/1990', periods=n, freq='H', tz='US/Eastern')
        ts = Series(np.random.randn(n), index=rng)
        return ts

    @pytest.mark.parametrize("index_val", [
        "1990-01-01 09:00:00+00:00",
        "1990-01-01 03:00:00-06:00",
        datetime(1990, 1, 1, 9, tzinfo=_tz('UTC')),
        tz('US/Central').localize(datetime(1990, 1, 1, 3))
    ])
    def test_datetime_setget_tz_pytz(self, tz_ts, index_val):
        # also test Timestamp tz handling, GH #2789
        result = tz_ts.copy()  # type: Series
        result[index_val] = 0
        result[index_val] = tz_ts[4]
        assert_series_equal(result, tz_ts)

    @pytest.fixture(scope='function')
    def tz_ts_ny(self):
        n = 50
        # testing with timezone, GH #2785
        rng = date_range('1/1/1990', periods=n, freq='H', tz='America/New_York')
        return Series(np.random.randn(n), index=rng)

    @pytest.mark.parametrize('index_val', [
        "1990-01-01 09:00:00+00:00",
        "1990-01-01 03:00:00-06:00",
        datetime(1990, 1, 1, 9, tzinfo=_tz('UTC')),
        datetime(1990, 1, 1, 3, tzinfo=_tz('America/Chicago'))
    ])
    def test_datetime_setget_tz_dateutil(self, index_val, tz_ts_ny):
        # also test Timestamp tz handling, GH #2789
        result = tz_ts_ny.copy()
        result[index_val] = 0
        result[index_val] = tz_ts_ny[4]
        assert_series_equal(result, tz_ts_ny)

    @pytest.mark.parametrize(('lb', 'rb'), [
        ("1990-01-01 04:00:00", "1990-01-01 07:00:00"),
        (datetime(1990, 1, 1, 4), datetime(1990, 1, 1, 7)),

    ])
    def test_datetime_setget_(self, tz_ts, lb, rb):
        result = tz_ts[lb]
        expected = tz_ts[4]
        assert result == expected

        result = tz_ts.copy()
        result[lb] = 0
        result[lb] = tz_ts[4]
        assert_series_equal(result, tz_ts)

        result = tz_ts[lb:rb]
        expected = tz_ts[4:8]
        assert_series_equal(result, expected)

        result = tz_ts.copy()
        result[lb:rb] = 0
        result[lb:rb] = tz_ts[4:8]
        assert_series_equal(result, tz_ts)

        # TODO: wrapping the datetimes in pd.Timestamp() fails here.
        result = tz_ts[(tz_ts.index >= lb) & (tz_ts.index <= rb)]
        expected = tz_ts[4:8]
        assert_series_equal(result, expected)

        result = tz_ts[tz_ts.index[4]]
        expected = tz_ts[4]
        assert result == expected

        result = tz_ts[tz_ts.index[4:8]]
        expected = tz_ts[4:8]
        assert_series_equal(result, expected)

        result = tz_ts.copy()
        result[tz_ts.index[4:8]] = 0
        result[4:8] = tz_ts[4:8]
        assert_series_equal(result, tz_ts)

        # also test partial date slicing
        result = tz_ts["1990-01-02"]
        expected = tz_ts[24:48]
        assert_series_equal(result, expected)

        result = tz_ts.copy()
        result["1990-01-02"] = 0
        result["1990-01-02"] = tz_ts[24:48]
        assert_series_equal(result, tz_ts)

    def test_datetime_setget_periodindex(self):

        N = 50
        rng = period_range('1/1/1990', periods=N, freq='H')
        ts = Series(np.random.randn(N), index=rng)

        result = ts["1990-01-01 04"]
        expected = ts[4]
        assert result == expected

        result = ts.copy()
        result["1990-01-01 04"] = 0
        result["1990-01-01 04"] = ts[4]
        assert_series_equal(result, ts)

        result = ts["1990-01-01 04":"1990-01-01 07"]
        expected = ts[4:8]
        assert_series_equal(result, expected)

        result = ts.copy()
        result["1990-01-01 04":"1990-01-01 07"] = 0
        result["1990-01-01 04":"1990-01-01 07"] = ts[4:8]
        assert_series_equal(result, ts)

        lb = "1990-01-01 04"
        rb = "1990-01-01 07"
        result = ts[(ts.index >= lb) & (ts.index <= rb)]
        expected = ts[4:8]
        assert_series_equal(result, expected)

        # GH 2782
        result = ts[ts.index[4]]
        expected = ts[4]
        assert result == expected

        result = ts[ts.index[4:8]]
        expected = ts[4:8]
        assert_series_equal(result, expected)

        result = ts.copy()
        result[ts.index[4:8]] = 0
        result[4:8] = ts[4:8]
        assert_series_equal(result, ts)


class TestDatetimeIndexing_Fancy_GetSet(object):

    @pytest.fixture(scope='session')
    def series(self):
        # DatetimeIndex of first Friday for each month 2005-2009
        dti = DatetimeIndex(freq='WOM-1FRI',
                                 start=datetime(2005, 1, 1),
                                 end=datetime(2010, 1, 1))

        return Series(np.arange(len(dti)), index=dti)

    @pytest.mark.parametrize('index_val', [
        48,
        '1/2/2009',
        '2009-1-2',
        datetime(2009, 1, 2),
        lib.Timestamp(datetime(2009, 1, 2))
    ])
    def test_fancy_getitem(self, index_val, series):
        assert series[index_val] == 48

    @pytest.mark.parametrize("date", [
        '2009-01-03',
        datetime(2009, 1, 3),
        '2009/1/3',
        '2009-1-3'
    ])
    def test_fancy_getitem_error(self, date, series):
        pytest.raises(KeyError, series.__getitem__, date)

    def test_fancy_getitem_slice(self, series):
        # two different str->date conversions
        slice_strings = series['3/6/2009':'2009-06-05']
        slice_datetime = series[datetime(2009, 3, 6):datetime(2009, 6, 5)]

        assert_series_equal(slice_strings, slice_datetime)

    @pytest.mark.parametrize(('index_val', 'expected'), [
        (48, -1),
        ('1/2/2009', -2),
        ('2009-01-02', -2),
        ('2009-1-2', -2)
    ])
    def test_fancy_setitem(self, index_val, expected, series):
        series[index_val] = expected
        assert series[48] == expected

    def test_fancy_setitem_slice(self, series):
        series['1/2/2009':'2009-06-05'] = -3
        assert all(series[48:54] == -3)


class TestTimeSeriesDuplicates(object):

    @pytest.fixture(scope='function')
    def dupes(self):
        dates = [datetime(2000, 1, 2),
                 datetime(2000, 1, 2),
                 datetime(2000, 1, 2),
                 datetime(2000, 1, 3),
                 datetime(2000, 1, 3),
                 datetime(2000, 1, 3),
                 datetime(2000, 1, 4),
                 datetime(2000, 1, 4),
                 datetime(2000, 1, 4),
                 datetime(2000, 1, 5)]

        return Series(np.random.randn(len(dates)), index=dates)

    def test_constructor(self, dupes):
        assert isinstance(dupes, Series)
        assert isinstance(dupes.index, DatetimeIndex)

    def test_is_unique_monotonic(self, dupes):
        assert not dupes.index.is_unique

    def test_index_unique(self, dupes):
        uniques = dupes.index.unique()
        expected = DatetimeIndex([datetime(2000, 1, 2), datetime(2000, 1, 3),
                                  datetime(2000, 1, 4), datetime(2000, 1, 5)])
        assert uniques.dtype == 'M8[ns]'  # sanity
        tm.assert_index_equal(uniques, expected)
        assert dupes.index.nunique() == 4

        # #2563
        assert isinstance(uniques, DatetimeIndex)

    def test_index_unique_tz_localize(self, dupes):

        expected = DatetimeIndex([datetime(2000, 1, 2), datetime(2000, 1, 3),
                                  datetime(2000, 1, 4), datetime(2000, 1, 5)])

        dups_local = dupes.index.tz_localize('US/Eastern')
        dups_local.name = 'foo'
        result = dups_local.unique()
        expected = DatetimeIndex(expected, name='foo')
        expected = expected.tz_localize('US/Eastern')
        assert result.tz is not None
        assert result.name == 'foo'
        tm.assert_index_equal(result, expected)

    def test_index_dupes_contains(self):
        d = datetime(2011, 12, 5, 20, 30)
        ix = DatetimeIndex([d, d])
        assert d in ix

    @pytest.mark.parametrize("date", [
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
        datetime(2000, 1, 5)
    ])
    def test_duplicate_dates_indexing(self, date, dupes):
        ts = dupes.copy()  # type: Series

        result = ts[date]
        mask = ts.index == date
        total = sum(mask)
        expected = ts[mask]
        if total > 1:
            assert_series_equal(result, expected)
        else:
            assert_almost_equal(result, expected[0])

        cp = ts.copy()  # type: Series
        cp[date] = 0
        expected = Series(np.where(mask, 0, dupes), index=dupes.index)
        assert_series_equal(cp, expected)

        pytest.raises(KeyError, dupes.__getitem__, datetime(2000, 1, 6))

        # new index
        ts[datetime(2000, 1, 6)] = 0
        assert ts[datetime(2000, 1, 6)] == 0

    def test_range_slice(self):
        idx = DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/3/2000',
                             '1/4/2000'])

        ts = Series(np.random.randn(len(idx)), index=idx)

        result = ts['1/2/2000':]
        expected = ts[1:]
        assert_series_equal(result, expected)

        result = ts['1/2/2000':'1/3/2000']
        expected = ts[1:4]
        assert_series_equal(result, expected)

    def test_groupby_average_dup_values(self, dupes):
        result = dupes.groupby(level=0).mean()
        expected = dupes.groupby(dupes.index).mean()
        assert_series_equal(result, expected)

    def test_indexing_over_size_cutoff(self):
        # #1821

        old_cutoff = _index._SIZE_CUTOFF
        try:
            _index._SIZE_CUTOFF = 1000

            # create large list of non periodic datetime
            dates = []
            sec = timedelta(seconds=1)
            half_sec = timedelta(microseconds=500000)
            d = datetime(2011, 12, 5, 20, 30)
            n = 1100
            for _ in range(n):
                dates.append(d)
                dates.append(d + sec)
                dates.append(d + sec + half_sec)
                dates.append(d + sec + sec + half_sec)
                d += 3 * sec

            # duplicate some values in the list
            duplicate_positions = np.random.randint(0, len(dates) - 1, 20)
            for p in duplicate_positions:
                dates[p + 1] = dates[p]

            df = DataFrame(np.random.randn(len(dates), 4),
                           index=dates,
                           columns=list('ABCD'))

            pos = n * 3
            timestamp = df.index[pos]
            assert timestamp in df.index

            assert len(df.loc[[timestamp]]) > 0
        finally:
            _index._SIZE_CUTOFF = old_cutoff

    def test_indexing_unordered(self):
        # GH 2437
        rng = pd.date_range(start='2011-01-01', end='2011-01-15')
        ts = Series(np.random.rand(len(rng)), index=rng)
        ts2 = pd.concat([ts[0:4], ts[-4:], ts[4:-4]])

        for t in ts.index:

            expected = ts[t]
            result = ts2[t]

            assert expected == result

        # GH 3448 (ranges)
        def compare(slobj):
            result = ts2[slobj].copy()
            result = result.sort_index()
            expected = ts[slobj]
            assert_series_equal(result, expected)

        compare(slice('2011-01-01', '2011-01-15'))
        compare(slice('2010-12-30', '2011-01-15'))
        compare(slice('2011-01-01', '2011-01-16'))

        # partial ranges
        compare(slice('2011-01-01', '2011-01-6'))
        compare(slice('2011-01-06', '2011-01-8'))
        compare(slice('2011-01-06', '2011-01-12'))

        # single values
        result = ts2['2011'].sort_index()
        expected = ts['2011']
        assert_series_equal(result, expected)

        # diff freq
        rng = pd.date_range(datetime(2005, 1, 1), periods=20, freq='M')
        ts = Series(np.arange(len(rng)), index=rng)
        ts = ts.take(np.random.permutation(20))

        result = ts['2005']
        for t in result.index:
            assert t.year == 2005

    @pytest.fixture(scope='function')
    def ts(self):
        idx = pd.date_range("2001-1-1", periods=20, freq='M')
        ts = Series(np.random.rand(len(idx)), index=idx)
        return ts

    def test_indexing_getting(self, ts):

        # GH 3070, make sure semantics work on Series/Frame
        expected = ts['2001']
        expected.name = 'A'

        df = DataFrame(dict(A=ts))
        result = df['2001']['A']
        assert_series_equal(expected, result)

    def test_indexing_setting(self, ts):

        df = DataFrame(dict(A=ts))

        ts['2001'] = 1
        expected = ts['2001']
        expected.name = 'A'

        df.loc['2001', 'A'] = 1

        result = df['2001']['A']
        assert_series_equal(expected, result)

    def test_indexing_last_day_inclusive_freq_hour(self):
        # GH3546 (not including times on the last day)
        idx = pd.date_range(start='2013-05-31 00:00', end='2013-05-31 23:00',
                            freq='H')
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013-05']
        assert_series_equal(expected, ts)

    def test_indexing_last_day_inclusive_freq_sec(self):

        idx = pd.date_range(start='2013-05-31 00:00', end='2013-05-31 23:59',
                            freq='S')
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013-05']
        assert_series_equal(expected, ts)

    def test_indexing_last_day_inclusive_freq_microsec(self):

        idx = [Timestamp('2013-05-31 00:00'),
               Timestamp(datetime(2013, 5, 31, 23, 59, 59, 999999))]
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013']
        assert_series_equal(expected, ts)

    def test_indexing_seconds_resolution(self):
        # GH14826, indexing with a seconds resolution string / datetime object
        df = DataFrame(np.random.rand(5, 5),
                       columns=['open', 'high', 'low', 'close', 'volume'],
                       index=pd.date_range('2012-01-02 18:01:00',
                                           periods=5, tz='US/Central', freq='s'))
        expected = df.loc[[df.index[2]]]
        assert isinstance(expected, DataFrame)
        assert len(expected) == 1

        # this is a single date, so will raise
        pytest.raises(KeyError, df.__getitem__, '2012-01-02 18:01:02', )
        pytest.raises(KeyError, df.__getitem__, df.index[2], )
