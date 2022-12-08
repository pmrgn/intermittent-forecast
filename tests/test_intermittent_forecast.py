import unittest
from intermittent_forecast import error_metrics, croston, Adida, Imapa
import numpy as np

class TestErrorMetrics(unittest.TestCase):

    def test_error_metrics(self):
        ts =  np.arange(6)
        f = np.insert(np.ones(5), 0, np.nan)
        self.assertAlmostEqual(error_metrics.mae(ts, f), 2)
        self.assertAlmostEqual(error_metrics.mse(ts, f), 6)
        self.assertAlmostEqual(error_metrics.msr(ts, f), 0.75)
        self.assertAlmostEqual(error_metrics.mar(ts, f), 0.7)
        self.assertAlmostEqual(error_metrics.pis(ts, f), 20)


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
        n_rpt = 5
        ts = [1,2,0,3] * n_rpt
        exp = np.array([6] * n_rpt, dtype='float')
        exp = np.insert(exp, 0, np.nan)
        adida = Adida(ts).agg(size=4, overlapping=False)
        pred_cro = adida.predict(croston, alpha=1, beta=1)
        self.assertIsNone(
            np.testing.assert_array_equal(pred_cro.prediction,exp)
        )
        pred_sba = adida.predict(croston, method='sba', alpha=1, beta=1)
        self.assertIsNone(
            np.testing.assert_array_equal(pred_cro.prediction,exp*0.5)
        )


    def test_disaggregation(self):
        # Aggregation size of 1 will produce the same forecast as using one of
        # the models in the croston function
        ts = np.random.randint(0,10,20)
        adida = (
            Adida(ts).agg(size=1, overlapping=False)
            .predict(croston, alpha=0.5)
            .disagg(h=1, cycle=None)
        )
        cro = croston(ts, alpha=0.5)
        self.assertIsNone(
            np.testing.assert_array_equal(adida, cro)
        )
        
        ts = [3,3,3,3,0,0,0,0] * 2
        adida = (
            Adida(ts).agg(size=4, overlapping=False)
            .predict(croston, method='cro', alpha=1)
            .disagg(h=1, cycle=None)
        )
        exp = np.append(
            np.repeat([np.nan,2,2,1.5],4),
            [1.5]
        )
        self.assertIsNone(
            np.testing.assert_array_equal(adida, exp)
        )
        
        # Seasonal Disaggregation
        ts = [1,2,3,4] * 4
        adida = (
            Adida(ts).agg(size=4, overlapping=False)
            .predict(croston, method='cro', alpha=1)
            .disagg(h=4, cycle=4)
        )
        exp = np.append([np.nan]*4,ts)
        self.assertIsNone(
            np.testing.assert_array_equal(adida, exp)
        )
        
        
    #     f2 = adida.disagg(h=4, cycle=4)
    #     exp2 = [0,1,2,3]
    #     self.assertIsNone(np.testing.assert_array_equal(f2,exp2))

    #     # Pass a prediction value straight to the disagg method
    #     f3 = (
    #         Adida(ts).agg(size=4, overlapping=False)
    #         .disagg(prediction=5, h=4, cycle=4)
    #     )
    #     exp3 = np.array([0,1,2,3]) * 5/6
    #     self.assertIsNone(np.testing.assert_allclose(f3,exp3))


# class TestImapa(unittest.TestCase):

#     def test_agg(self):
#         ts = [0,0,0,10,0,0,0,2,0,0,0,3]
#         sizes = [1,2]
#         imapa = Imapa(ts).agg(sizes)
#         for agg in imapa.aggregated:
#             print(agg)
#         f = imapa.predict(combine='mean')
#         print(f)
                
if __name__ == '__main__':
    unittest.main()
                             