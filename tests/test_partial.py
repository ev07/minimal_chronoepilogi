import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from chronoepilogi import partial





#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

def test_temporal_slow_partial_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
    variable_types = dict([(column, "numerical") for column in data.columns])
    asso = partial.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = asso.partial_corr(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["1"]]),pd.DataFrame(data[["2"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.07208953, 0.05627934],[0.03686298, 0.04137501]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.09649547, 0.10936624],[0.02173326, 0.03464236]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.07455153, 0.03384917]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.09990165, 0.02838155]), atol=1e-8)

def test_temporal_slow_partial_mixed():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
    categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
    asso = partial.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = asso.partial_corr(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["1"]]),pd.DataFrame(data[["3"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.04385652, 0.0345677 ],[0.03809128, 0.02866959]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.39990684, 0.43878687], [0.06797133, 0.07284824]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.04112653, 0.03111284]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.21688858, 0.03302831]), atol=1e-8)

def test_temporal_slow_partial_mixed_kruskal():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
    categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
    asso = partial.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"kruskal","variable_types":variable_types})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = asso.partial_corr(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["1"]]),pd.DataFrame(data[["3"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.04385652, 0.0345677 ],[0.03809128, 0.02866959]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.39990684, 0.43878687], [0.06797133, 0.07284824]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.04112653, 0.03111284]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.21501478, 0.03525033]), atol=1e-8)

def test_temporal_slow_partial_mixed_alexandergovern():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
    categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
    asso = partial.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"alexandergovern","variable_types":variable_types})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = asso.partial_corr(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["1"]]),pd.DataFrame(data[["3"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.04385652, 0.0345677 ],[0.03809128, 0.02866959]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.39990684, 0.43878687], [0.06797133, 0.07284824]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.04112653, 0.03111284]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.22347802, 0.02917972]), atol=1e-8)


def test_crosssectional_partial_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,7)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G1","c"),("G2","a"),("G2","b"),("G2","c")]))
    variable_types = dict([(group, "numerical") for group in data.columns.get_level_values(0).unique()])
    parcorr = partial.CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = parcorr.partial_corr(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1"]]), pd.DataFrame(data[["G2"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.36888558, 0.33352269],[0.0014652 , 0.0013898 ]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.2173927 , 0.25930918],[0.07060479, 0.08380666]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.3808198 , 0.00130095]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.22332839, 0.07809404]), atol=1e-8)

def test_crosssectional_partial_mixed():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,4)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b"),("G1","c")]))
    categorical = pd.DataFrame(rng.integers(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
    parcorr =  partial.CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = parcorr.partial_corr(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1"]]), pd.DataFrame(data[["G2"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.59614113, 0.59506099],[0.03118282, 0.03196968]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.73737096, 0.70131846],[0.60607526, 0.57785402]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.57745635, 0.03535778]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.5959402 , 0.45854079]), atol=1e-8)
    

#####################################################
#                                                   #
#                Machine-generated tests            #
#                                                   #
#####################################################

# TemporalLinearRIT

def test_temporal_linear_rit_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(300,3)),columns=["target","1","2"])
    rit = partial.TemporalLinearRIT({"lags":2})
    pvalue = rit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1"]]), pd.DataFrame(data[["2"]]))
    assert 0.0 <= pvalue <= 1.0
    npt.assert_allclose(pvalue, 0.0065474, atol=1e-4)

def test_temporal_linear_rit_detects_dependence():
    rng = np.random.default_rng(1)
    n = 300
    candidate = rng.random(n)
    condition = rng.random(n)
    target = np.zeros(n)
    target[1:] = 0.8*candidate[:-1] + 0.05*rng.random(n-1)
    data = pd.DataFrame({"target":target,"cand":candidate,"cond":condition})
    rit = partial.TemporalLinearRIT({"lags":2})
    pvalue = rit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["cand"]]), pd.DataFrame(data[["cond"]]))
    assert pvalue < 1e-3

def test_temporal_linear_rit_redundant_when_collinear():
    rng = np.random.default_rng(5)
    n = 300
    condition = rng.random(n)
    candidate = condition + 0.01*rng.normal(size=n)
    target = np.zeros(n)
    target[1:] = 0.8*condition[:-1] + 0.3*rng.normal(size=n-1)
    data = pd.DataFrame({"target":target,"cand":candidate,"cond":condition})
    rit = partial.TemporalLinearRIT({"lags":2})
    pvalue = rit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["cand"]]), pd.DataFrame(data[["cond"]]))
    assert pvalue > 0.05

def test_temporal_linear_rit_missing_lags_raises():
    with pytest.raises(AssertionError):
        partial.TemporalLinearRIT({})

# CrossSectionalLinearRIT

def test_crosssectional_linear_rit_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(500,7)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G1","c"),("G2","a"),("G2","b"),("G2","c")]))
    crit = partial.CrossSectionalLinearRIT({"large_sample":False})
    pvalue = crit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1"]]), pd.DataFrame(data[["G2"]]))
    assert 0.0 <= pvalue <= 1.0
    npt.assert_allclose(pvalue, 4.74620381873348e-07, atol=1e-8)

def test_crosssectional_linear_rit_detects_dependence():
    rng = np.random.default_rng(2)
    n = 500
    g1a = rng.random(n); g1b = rng.random(n)
    g2a = rng.random(n); g2b = rng.random(n)
    target = 0.9*g1a + 0.9*g1b + 0.05*rng.random(n)
    data = pd.DataFrame({"target":target,"g1a":g1a,"g1b":g1b,"g2a":g2a,"g2b":g2b})
    crit = partial.CrossSectionalLinearRIT({"large_sample":False})
    pvalue = crit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["g1a","g1b"]]), pd.DataFrame(data[["g2a","g2b"]]))
    assert pvalue < 1e-6

def test_crosssectional_linear_rit_redundant_when_collinear():
    rng = np.random.default_rng(4)
    n = 500
    g2a = rng.random(n); g2b = rng.random(n)
    g1a = g2a + 0.01*rng.normal(size=n)
    g1b = g2b + 0.01*rng.normal(size=n)
    target = 0.9*g2a + 0.9*g2b + 0.5*rng.normal(size=n)
    data = pd.DataFrame({"target":target,"g1a":g1a,"g1b":g1b,"g2a":g2a,"g2b":g2b})
    crit = partial.CrossSectionalLinearRIT({"large_sample":False})
    pvalue = crit.ci_test(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["g1a","g1b"]]), pd.DataFrame(data[["g2a","g2b"]]))
    assert pvalue > 0.05

def test_crosssectional_linear_rit_missing_large_sample_raises():
    with pytest.raises(AssertionError):
        partial.CrossSectionalLinearRIT({})