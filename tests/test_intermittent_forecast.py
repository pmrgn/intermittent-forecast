import unittest
from intermittent_forecast import _error_metrics, croston, adida, imapa
from intermittent_forecast.adida import (_aggregate, _apply_cycle_perc, 
                                         _seasonal_cycle)
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


class TestErrorMetrics(unittest.TestCase):

    def test_error_metrics(self):
        ts =  np.arange(6)
        f = np.insert(np.ones(5), 0, np.nan)
        self.assertAlmostEqual(_error_metrics.mae(ts, f), 2)
        self.assertAlmostEqual(_error_metrics.mse(ts, f), 6)
        self.assertAlmostEqual(_error_metrics.msr(ts, f), 0.75)
        self.assertAlmostEqual(_error_metrics.mar(ts, f), 0.7)
        self.assertAlmostEqual(_error_metrics.pis(ts, f), 20)


class TestCroston(unittest.TestCase):

    def test_croston(self):
        a = 0.5
        b = 0.4
        ts = [0,1,0,2]
        expected = {
            'cro': np.array([np.nan, np.nan, 0.5, 0.5, 0.75]),
            'sba': np.array([np.nan, np.nan, 0.4, 0.4, 0.6]),
            'sbj': np.array([np.nan, np.nan, 0.375, 0.375, 0.5625]),
            'tsb': np.array([np.nan, 0.5, 0.7, 0.42, 0.978])
        }
        for k, v in expected.items():
            f = croston(ts, method=k, alpha=a, beta=b, opt=False)
            self.assertIsNone(assert_allclose(v,f))


class TestAdida(unittest.TestCase):
    
    def test_aggregate(self):
        ts = np.array([0,1,2,0,3])
        agg = _aggregate(ts, size=1, overlapping=False)
        self.assertIsNone(assert_array_equal(agg, ts))

        agg = _aggregate(ts, size=2, overlapping=False)
        self.assertIsNone(assert_array_equal(agg, [3,3]))

        agg = _aggregate(ts, size=2, overlapping=True)
        self.assertIsNone(assert_array_equal(agg, [1,3,2,3])) 

    def test_seasonal_cycle(self):
        ts = np.tile(np.arange(5), 3).astype('float')
        s = _seasonal_cycle(ts, cycle=5)
        self.assertIsNone(
            assert_allclose(s, [0, .1, .2, .3, .4])
            ) 

        ts = np.array([1,0,1,1,0,1,1])
        s = _seasonal_cycle(ts, cycle=3)
        self.assertIsNone(
            assert_allclose(s, [0, .5, .5])
            ) 

    def test_apply_cycle_perc(self):
        f = _apply_cycle_perc(np.ones(5), [0, .3, .7])
        self.assertIsNone(
            assert_allclose(f, [.3, .7, 0, .3, .7])
            ) 

    def test_adida(self):
        ts = np.arange(1,11)
        f = adida(ts, size=1, overlapping=False, method='cro',
                  alpha=1, h=1)
        expected = np.insert(ts.astype('float'), 0, [np.nan])
        self.assertIsNone(assert_array_equal(f, expected))

        f = adida(ts, size=2, overlapping=False, method='cro',
                  alpha=1, h=2)
        expected = np.array([np.nan, 1.5, 3.5, 5.5, 7.5, 9.5]).repeat(2)
        self.assertIsNone(assert_array_equal(f, expected))
        
        f = adida(ts, size=2, overlapping=True, method='cro',
                  alpha=1, h=1)
        expected = np.concatenate(([np.nan, np.nan], np.arange(1.5, 10.5, 1)))
        self.assertIsNone(assert_array_equal(f, expected))

        # Test for seasonal cycles
        ts = np.tile(np.arange(7), 5)
        f = adida(ts, size=7, overlapping=False, method='sba',
                  alpha=1, h=7, cycle=7)
        expected = np.insert(ts/2, 0, [np.nan]*7)
        self.assertIsNone(assert_array_equal(f, expected))


class TestImapa(unittest.TestCase):
    
    def test_imapa(self):
        ts = np.arange(1,13)
        
        # IMAPA at single aggregation size should equal ADIDA
        for i in range(1,5):
            f = imapa(ts, sizes=[i])
            expected = adida(ts, size=i)
            self.assertIsNone(assert_allclose(f, expected))
        
        # Test mean combination
        f = imapa(ts, sizes=[1,2,3], overlapping=False, method='cro',
                  alpha=1, h=1, combine='mean')
        expected = (1/3) * np.array([np.nan, np.nan, np.nan, 6.5, 9.5, 10.5,
                             16.5, 17.5, 20.5, 24.5, 27.5, 28.5, 34.5])
        self.assertIsNone(assert_array_equal(f, expected))

        # Test median combination
        f = imapa(ts, sizes=[1,2,3], overlapping=False, method='sba',
                  alpha=1, h=1, combine='median')
        expected = np.array([np.nan, np.nan, np.nan, 1, 1.75, 1.75,
                             2.75, 2.75, 3.75, 4, 4.75, 4.75, 5.75])
        self.assertIsNone(assert_array_equal(f, expected))


if __name__ == '__main__':
    unittest.main()