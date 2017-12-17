# coding=utf-8
# pylint: disable-msg=E1101,W0612
from datetime import datetime, timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import Series, DatetimeIndex, Timestamp, NaT, DataFrame, date_range
from pandas._libs import tslib, index as _index
from pandas.compat import range, lrange
from pandas.util import testing as tm
from pandas.util.testing import assert_series_equal, assert_almost_equal


class TestTimeSeriesDuplicates(object):

    def setup_method(self, method):
        dates = [datetime(2000, 1, 2), datetime(2000, 1, 2),
                 datetime(2000, 1, 2), datetime(2000, 1, 3),
                 datetime(2000, 1, 3), datetime(2000, 1, 3),
                 datetime(2000, 1, 4), datetime(2000, 1, 4),
                 datetime(2000, 1, 4), datetime(2000, 1, 5)]

        self.dups = Series(np.random.randn(len(dates)), index=dates)

    def test_constructor(self):
        assert isinstance(self.dups, Series)
        assert isinstance(self.dups.index, DatetimeIndex)

    def test_is_unique_monotonic(self):
        assert not self.dups.index.is_unique

    def test_index_unique(self):
        uniques = self.dups.index.unique()
        expected = DatetimeIndex([datetime(2000, 1, 2), datetime(2000, 1, 3),
                                  datetime(2000, 1, 4), datetime(2000, 1, 5)])
        assert uniques.dtype == 'M8[ns]'  # sanity
        tm.assert_index_equal(uniques, expected)
        assert self.dups.index.nunique() == 4

        # #2563
        assert isinstance(uniques, DatetimeIndex)

        dups_local = self.dups.index.tz_localize('US/Eastern')
        dups_local.name = 'foo'
        result = dups_local.unique()
        expected = DatetimeIndex(expected, name='foo')
        expected = expected.tz_localize('US/Eastern')
        assert result.tz is not None
        assert result.name == 'foo'
        tm.assert_index_equal(result, expected)

        # NaT, note this is excluded
        arr = [1370745748 + t for t in range(20)] + [tslib.iNaT]
        idx = DatetimeIndex(arr * 3)
        tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
        assert idx.nunique() == 20
        assert idx.nunique(dropna=False) == 21

        arr = [Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t)
               for t in range(20)] + [NaT]
        idx = DatetimeIndex(arr * 3)
        tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
        assert idx.nunique() == 20
        assert idx.nunique(dropna=False) == 21

    def test_index_dupes_contains(self):
        d = datetime(2011, 12, 5, 20, 30)
        ix = DatetimeIndex([d, d])
        assert d in ix

    def test_duplicate_dates_indexing(self):
        ts = self.dups

        uniques = ts.index.unique()
        for date in uniques:
            result = ts[date]

            mask = ts.index == date
            total = (ts.index == date).sum()
            expected = ts[mask]
            if total > 1:
                assert_series_equal(result, expected)
            else:
                assert_almost_equal(result, expected[0])

            cp = ts.copy()
            cp[date] = 0
            expected = Series(np.where(mask, 0, ts), index=ts.index)
            assert_series_equal(cp, expected)

        pytest.raises(KeyError, ts.__getitem__, datetime(2000, 1, 6))

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

    def test_groupby_average_dup_values(self):
        result = self.dups.groupby(level=0).mean()
        expected = self.dups.groupby(self.dups.index).mean()
        assert_series_equal(result, expected)

    def test_indexing_over_size_cutoff(self):
        import datetime
        # #1821

        old_cutoff = _index._SIZE_CUTOFF
        try:
            _index._SIZE_CUTOFF = 1000

            # create large list of non periodic datetime
            dates = []
            sec = datetime.timedelta(seconds=1)
            half_sec = datetime.timedelta(microseconds=500000)
            d = datetime.datetime(2011, 12, 5, 20, 30)
            n = 1100
            for i in range(n):
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

            # it works!
            df.loc[timestamp]
            assert len(df.loc[[timestamp]]) > 0
        finally:
            _index._SIZE_CUTOFF = old_cutoff

    def test_indexing_unordered(self):
        # GH 2437
        rng = date_range(start='2011-01-01', end='2011-01-15')
        ts = Series(np.random.rand(len(rng)), index=rng)
        ts2 = pd.concat([ts[0:4], ts[-4:], ts[4:-4]])

        for t in ts.index:
            # TODO: unused?
            s = str(t)  # noqa

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
        rng = date_range(datetime(2005, 1, 1), periods=20, freq='M')
        ts = Series(np.arange(len(rng)), index=rng)
        ts = ts.take(np.random.permutation(20))

        result = ts['2005']
        for t in result.index:
            assert t.year == 2005

    def test_indexing(self):

        idx = date_range("2001-1-1", periods=20, freq='M')
        ts = Series(np.random.rand(len(idx)), index=idx)

        # getting

        # GH 3070, make sure semantics work on Series/Frame
        expected = ts['2001']
        expected.name = 'A'

        df = DataFrame(dict(A=ts))
        result = df['2001']['A']
        assert_series_equal(expected, result)

        # setting
        ts['2001'] = 1
        expected = ts['2001']
        expected.name = 'A'

        df.loc['2001', 'A'] = 1

        result = df['2001']['A']
        assert_series_equal(expected, result)

        # GH3546 (not including times on the last day)
        idx = date_range(start='2013-05-31 00:00', end='2013-05-31 23:00',
                         freq='H')
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013-05']
        assert_series_equal(expected, ts)

        idx = date_range(start='2013-05-31 00:00', end='2013-05-31 23:59',
                         freq='S')
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013-05']
        assert_series_equal(expected, ts)

        idx = [Timestamp('2013-05-31 00:00'),
               Timestamp(datetime(2013, 5, 31, 23, 59, 59, 999999))]
        ts = Series(lrange(len(idx)), index=idx)
        expected = ts['2013']
        assert_series_equal(expected, ts)

        # GH14826, indexing with a seconds resolution string / datetime object
        df = DataFrame(np.random.rand(5, 5),
                       columns=['open', 'high', 'low', 'close', 'volume'],
                       index=date_range('2012-01-02 18:01:00',
                                        periods=5, tz='US/Central', freq='s'))
        expected = df.loc[[df.index[2]]]

        # this is a single date, so will raise
        pytest.raises(KeyError, df.__getitem__, '2012-01-02 18:01:02', )
        pytest.raises(KeyError, df.__getitem__, df.index[2], )
