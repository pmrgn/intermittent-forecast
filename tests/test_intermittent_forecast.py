import unittest
from intermittent_forecast import error_metrics, croston, Adida
import numpy as np

class TestErrorMetrics(unittest.TestCase):

    def test_error_metrics(self):
        # Tests for error metrics
        arr1 =  np.arange(5)
        arr2 = np.ones(5)
        self.assertAlmostEqual(error_metrics.mae(arr1,arr2), 1.4)
        self.assertAlmostEqual(error_metrics.mse(arr1,arr2), 3)
        self.assertAlmostEqual(error_metrics.msr(arr1,arr2), 0.25)
        self.assertAlmostEqual(error_metrics.pis(arr1,arr2), 5)


class TestCroston(unittest.TestCase):

    def test_croston(self):
        # Tests for forecasts
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
            self.assertIsNone(np.testing.assert_allclose(v,f))


class TestAdida(unittest.TestCase):

    def test_adida(self):
        ts = [1,2,3] * 10
        f = (Adida(ts)
             .agg(size=3, overlapping=False)
             .predict(croston, alpha=1, beta=1)
             .disagg(h=3, cycle=3)
        )
        self.assertIsNone(np.testing.assert_array_equal(f,ts[:3]))


if __name__ == '__main__':
    unittest.main()
                             
