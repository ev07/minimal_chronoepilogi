import numpy as np
import numpy.testing as npt
import pandas as pd

from chronoepilogi import associations





#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

def _make_temporal_numerical_data():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
    variable_types = dict([(column, "numerical") for column in data.columns])
    return data, variable_types

def _make_crosssectional_numerical_data():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(1000,5)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G2","a"),("G2","b")]))
    variable_types = dict([(column, "numerical") for column in data.columns.get_level_values(0).unique()])
    return data, variable_types

def _make_temporal_mixed_data():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
    categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
    return data, variable_types

def _make_crosssectional_mixed_data():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b")]))
    categorical = pd.DataFrame(rng.integers(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
    return data, variable_types

# Generical calls

def test_temporal_slow_association_numerical():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03384917, -0.02838155, -0.0633841 , -0.15107386]), atol=1e-8)

def test_temporal_slow_association_mixed():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03302831, -0.02551908]), atol=1e-8)

def test_crosssectional_association_numerical():
    data, variable_types = _make_crosssectional_numerical_data()
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    npt.assert_allclose(result, np.array([-0.32736175, -0.11320393]), atol=1e-8)


def test_crosssectional_association_mixed():
    data, variable_types = _make_crosssectional_mixed_data()
    asso =  associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["G1","G2"]]))
    npt.assert_allclose(result, np.array([-0.05543262, -0.0992026]), atol=1e-8)

# Verify that categorical tests option work

def test_temporal_slow_association_mixed_kruskal():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"kruskal","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03525033, -0.02783337]), atol=1e-8)

def test_crosssectional_association_mixed_kruskal():
    data, variable_types = _make_crosssectional_mixed_data()
    asso =  associations.CrossSectionalAssociation({"categorical_method":"kruskal","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["G1","G2"]]))
    npt.assert_allclose(result, np.array([-0.05543262, -0.10252213]), atol=1e-8)

def test_temporal_slow_association_mixed_alexandergovern():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"alexandergovern","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.02917972, -0.02495522]), atol=1e-8)

def test_crosssectional_association_mixed_alexandergovern():
    data, variable_types = _make_crosssectional_mixed_data()
    asso =  associations.CrossSectionalAssociation({"categorical_method":"alexandergovern","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]),pd.DataFrame(data[["G1","G2"]]))
    npt.assert_allclose(result, np.array([-0.05543262, -0.09315233]), atol=1e-8)

