import numpy as np
import numpy.testing as npt
import pandas as pd

from chronoepilogi import associations





#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

# data providers

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

def _make_temporal_npinteger_data():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame((10*rng.random(size=(1000,3))).astype(np.int64),columns=["target","1","2"])
    categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)).astype(np.int64),columns=["3","4"])
    data = pd.concat([numerical,categorical], axis="columns")
    variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
    return data, variable_types

def _make_crosssectional_npinteger_data():
    rng = np.random.default_rng(0)
    numerical = pd.DataFrame((5*rng.random(size=(1000,3))).astype(np.int64),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b")]))
    categorical = pd.DataFrame(rng.integers(0,5,size=(1000,3)).astype(np.int64),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
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

# Verify data dimensionality edge cases

def test_temporal_single_lag():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.TemporalSlowAssociation({"lags":1,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))

def test_crosssectional_single_groupsize():
    data, variable_types = _make_crosssectional_numerical_data()
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[[("G1","a"),("G2","a")]]))

def test_temporal_single_ts_lag100():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.TemporalSlowAssociation({"lags":100,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1"]]))

def test_temporal_single_ts_lag1():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.TemporalSlowAssociation({"lags":1,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1"]]))

def test_crosssectional_single_group():
    data, variable_types = _make_crosssectional_numerical_data()
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1"]]))

def test_temporal_two_observation():
    data, variable_types = _make_temporal_numerical_data()
    data = data.iloc[:3]
    asso = associations.TemporalSlowAssociation({"lags":1,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))

def test_temporal_102_observations_100lags():
    data, variable_types = _make_temporal_numerical_data()
    data = data.iloc[:102]
    asso = associations.TemporalSlowAssociation({"lags":100,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))

def test_crosssectional_two_observation():
    data, variable_types = _make_crosssectional_numerical_data()
    data = data.iloc[:2]
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))

# constant data: prevent nans from occuring
# to be added in a later version. Requires cleaning associations.py
# need to add cases for categorical data, currently, only test for numerical data is implemented.

def test_temporal_constant_data():
    data, variable_types = _make_temporal_numerical_data()
    data["1"] = 0
    asso = associations.TemporalSlowAssociation({"lags":1,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":100,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))
    assert not np.any(np.isnan(result))

def test_temporal_constant_data_categorical():
    data, variable_types = _make_temporal_mixed_data()
    data["3"] = 0
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"kruskal","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"alexandergovern","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))

def test_temporal_constant_residuals():
    data, variable_types = _make_temporal_numerical_data()
    data["target"] = 0
    asso = associations.TemporalSlowAssociation({"lags":1,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":100,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3"]]))
    assert not np.any(np.isnan(result))

def test_temporal_constant_residuals_categorical():
    data, variable_types = _make_temporal_mixed_data()
    data["target"] = 0
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"kruskal","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"alexandergovern","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    assert not np.any(np.isnan(result))

#def test_crosssectional_constant_data():
#    data, variable_types = _make_crosssectional_numerical_data()
#    data[("G1","a")] = 0
#    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
#    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
#    assert not np.any(np.isnan(result))
#
#def test_crosssectional_constant_residuals():
#    data, variable_types = _make_crosssectional_numerical_data()
#    data[("target","")] = 0
#    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
#    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
#    assert not np.any(np.isnan(result))


# input types: cannot throw an error due to types

def test_temporal_npinteger():
    data, variable_types = _make_temporal_npinteger_data()
    asso = associations.TemporalSlowAssociation({"lags":2,"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))

def test_crosssectional_npinteger():
    data, variable_types = _make_crosssectional_npinteger_data()
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    _ = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))

# parallelism: make sure n_jobs does not crash the associations

def test_temporal_slow_association_mixed_1job():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"n_jobs":1})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03302831, -0.02551908]), atol=1e-8)

def test_temporal_slow_association_mixed_m1job():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"n_jobs":-1})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03302831, -0.02551908]), atol=1e-8)

def test_temporal_slow_association_mixed_3job():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"n_jobs":3})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03302831, -0.02551908]), atol=1e-8)

def test_temporal_slow_association_mixed_m3job():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"n_jobs":-3})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03111284, -0.04568282, -0.03302831, -0.02551908]), atol=1e-8)

# spearman correlation

def test_temporal_slow_association_mixed_spearman():
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"numerical_method":"spearman","categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03071761, -0.0465202,  -0.03302831, -0.02551908]), atol=1e-8)