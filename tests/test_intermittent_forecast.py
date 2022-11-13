import unittest
import intermittent_forecast as intmt
import numpy as np

class TestWrmsse(unittest.TestCase):

    def test_intmt_croston(self):
        # Tests for error metrics
        arr1 =  np.arange(5)
        arr2 = np.ones(5)
        self.assertAlmostEqual(intmt.croston.mae(arr1,arr2), 1.4)
        self.assertAlmostEqual(intmt.croston.mse(arr1,arr2), 3)
        self.assertAlmostEqual(intmt.croston.msr(arr1,arr2), 0.25)
        self.assertAlmostEqual(intmt.croston.pis(arr1,arr2), 5)

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
            f = intmt.croston.croston(ts, method=k, alpha=a, beta=b, 
                                      opt=False)
            self.assertIsNone(np.testing.assert_allclose(v,f))


if __name__ == '__main__':
    unittest.main()
                             
