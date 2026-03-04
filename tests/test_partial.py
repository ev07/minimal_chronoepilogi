import numpy as np
import numpy.testing as npt
import pandas as pd

from chronoepilogi import associations





#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

def test_temporal_slow_partial_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
    variable_types = dict([(column, "numerical") for column in data.columns])
    asso = associations.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
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
    asso = associations.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
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
    asso = associations.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"kruskal","variable_types":variable_types})
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
    asso = associations.TemporalSlowHk({"lags":10,"k":2,"categorical_method":"alexandergovern","variable_types":variable_types})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = asso.partial_corr(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["1"]]),pd.DataFrame(data[["3"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.04385652, 0.0345677 ],[0.03809128, 0.02866959]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.39990684, 0.43878687], [0.06797133, 0.07284824]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.04112653, 0.03111284]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.22347802, 0.02917972]), atol=1e-8)


def test_crosssectional_partial_numerical():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,7)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G1","c"),("G2","a"),("G2","b"),("G2","c")]))
    variable_types = dict([(group, "numerical") for group in data.columns.get_level_values(0).unique()])
    parcorr = associations.CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
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
    parcorr =  associations.CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = parcorr.partial_corr(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1"]]), pd.DataFrame(data[["G2"]]))
    npt.assert_allclose(p_RCa_Co, np.array([[0.59614113, 0.59506099],[0.03118282, 0.03196968]]), atol=1e-8)
    npt.assert_allclose(p_RCo_Ca, np.array([[0.73737096, 0.70131846],[0.60607526, 0.57785402]]), atol=1e-8)
    npt.assert_allclose(p_RCa, np.array([0.57745635, 0.03535778]), atol=1e-8)
    npt.assert_allclose(p_RCo, np.array([0.5959402 , 0.45854079]), atol=1e-8)
    
        