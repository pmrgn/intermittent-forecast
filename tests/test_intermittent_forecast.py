import unittest
from intermittent_forecast import error_metrics, croston, Adida
import numpy as np

class TestErrorMetrics(unittest.TestCase):

    def test_error_metrics(self):
        arr1 =  np.arange(5)
        arr2 = np.ones(5)
        self.assertAlmostEqual(error_metrics.mae(arr1,arr2), 1.4)
        self.assertAlmostEqual(error_metrics.mse(arr1,arr2), 3)
        self.assertAlmostEqual(error_metrics.msr(arr1,arr2), 0.25)
        self.assertAlmostEqual(error_metrics.pis(arr1,arr2), 5)


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
            self.assertIsNone(np.testing.assert_allclose(v,f))


class TestAdida(unittest.TestCase):
    
    def test_aggregation(self):
        ts = [0,1,2,0,3]
        agg1 = Adida(ts).agg(size=1).aggregated
        self.assertIsNone(np.testing.assert_array_equal(agg1,ts))

        agg2 = Adida(ts).agg(size=2, overlapping=False).aggregated
        exp2 = [3,3]
        self.assertIsNone(np.testing.assert_array_equal(agg2,exp2))

        agg3 = Adida(ts).agg(size=2, overlapping=True).aggregated
        exp3 = [1,3,2,3]
        self.assertIsNone(np.testing.assert_array_equal(agg3,exp3)) 


    def test_prediction(self):
        ts = [1,2,0,3] * 5
        adida = Adida(ts).agg(size=4, overlapping=False)
        pred_cro = adida.predict(croston, alpha=1, beta=1)
        self.assertEqual(pred_cro.prediction,6)

        pred_sba = adida.predict(croston, method='sba', alpha=1, beta=1)
        self.assertEqual(pred_sba.prediction,3)


    def test_disaggregation(self):
        ts = [0,1,2,3] * 5
        adida = (
            Adida(ts).agg(size=4, overlapping=False)
            .predict(croston, alpha=1, beta=1)
        )
        f1 = adida.disagg(h=1, cycle=None)
        exp1 = 6/4
        self.assertAlmostEqual(f1,exp1)

        f2 = adida.disagg(h=4, cycle=4)
        exp2 = [0,1,2,3]
        self.assertIsNone(np.testing.assert_array_equal(f2,exp2))

        # Pass a prediction value straight to the disagg method
        f3 = (
            Adida(ts).agg(size=4, overlapping=False)
            .disagg(prediction=5, h=4, cycle=4)
        )
        exp3 = np.array([0,1,2,3]) * 5/6
        self.assertIsNone(np.testing.assert_allclose(f3,exp3))


if __name__ == '__main__':
    unittest.main()
                             
