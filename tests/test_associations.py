import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from joblib import cpu_count

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

def test_temporal_slow_association_mixed_interleaved_columns():
    # same data and config as test_temporal_slow_association_mixed, but with
    # numerical and categorical columns interleaved instead of grouped by type,
    # to guard the index_num/index_cat bookkeeping in TemporalSlowAssociation.association().
    data, variable_types = _make_temporal_mixed_data()
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["3","1","4","2"]]))
    # expected values are the per-column results from test_temporal_slow_association_mixed,
    # reordered to match columns ["3","1","4","2"]:
    # "1"->-0.03111284, "2"->-0.04568282, "3"->-0.03302831, "4"->-0.02551908
    npt.assert_allclose(result, np.array([-0.03302831, -0.03111284, -0.02551908, -0.04568282]), atol=1e-8)

def test_temporal_unknown_variable_type_raises():
    data, variable_types = _make_temporal_mixed_data()
    variable_types["3"] = "unknown_type"
    asso = associations.TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
    with pytest.raises(ValueError):
        asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))

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

def test_crosssectional_constant_data():
    data, variable_types = _make_crosssectional_numerical_data()
    data[("G1","a")] = 0
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    assert not np.any(np.isnan(result))

def test_crosssectional_constant_residuals():
    data, variable_types = _make_crosssectional_numerical_data()
    data[("target","")] = 0
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    assert not np.any(np.isnan(result))

def test_crosssectional_constant_data_mixed():
    data, variable_types = _make_crosssectional_mixed_data()
    data[("G2","a")] = 0
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    assert not np.any(np.isnan(result))

def test_crosssectional_constant_residuals_mixed():
    data, variable_types = _make_crosssectional_mixed_data()
    data[("target","")] = 0
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    assert not np.any(np.isnan(result))

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
    asso = associations.TemporalSlowAssociation({"lags":10,"numerical_method":"spearmanr","categorical_method":"f_oneway","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["1","2","3","4"]]))
    npt.assert_allclose(result, np.array([-0.03071761, -0.0465202,  -0.03302831, -0.02551908]), atol=1e-8)

def test_crosssectional_mixed_spearman():
    data, variable_types = _make_crosssectional_mixed_data()
    asso = associations.CrossSectionalAssociation({"categorical_method":"f_oneway","numerical_method":"spearmanr","variable_types":variable_types})
    result = asso.association(pd.DataFrame(data[["target"]]), pd.DataFrame(data[["G1","G2"]]))
    npt.assert_allclose(result, np.array([-0.05521106, -0.0992026]), atol=1e-8)


#########################################################################################
#                                                                                       #
#                                  AI GENERATED TESTS                                   #
#                                                                                       #
#########################################################################################

#####################################################
#                                                   #
#            LaggedAssociationBase tests            #
#                                                   #
#####################################################

# A. ABC enforcement

def test_laggedassociationbase_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        associations.LaggedAssociationBase({"lags": 1})

def test_laggedassociationbase_subclass_without_apply_independence_tests_cannot_instantiate():
    class IncompleteAssociation(associations.LaggedAssociationBase):
        def association(self, residuals_df, variables_df):
            pass
    with pytest.raises(TypeError):
        IncompleteAssociation({"lags": 1})

def test_apply_independence_tests_is_abstract_on_base():
    method = associations.LaggedAssociationBase._apply_independence_tests
    assert getattr(method, "__isabstractmethod__", False)

# B. _cpus_from_njobs

def test_cpus_from_njobs_positive():
    asso = associations.PearsonMultivariate({"lags": 1})
    assert asso._cpus_from_njobs(3) == 3

def test_cpus_from_njobs_negative_all():
    asso = associations.PearsonMultivariate({"lags": 1})
    assert asso._cpus_from_njobs(-1) == cpu_count()

def test_cpus_from_njobs_negative_all_but_one():
    asso = associations.PearsonMultivariate({"lags": 1})
    assert asso._cpus_from_njobs(-2) == cpu_count() - 1

def test_cpus_from_njobs_zero_raises():
    asso = associations.PearsonMultivariate({"lags": 1})
    with pytest.raises(ValueError):
        asso._cpus_from_njobs(0)

# C. _check_inputs

def test_check_inputs_multi_column_residuals_raises():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types})
    with pytest.raises(ValueError):
        asso._check_inputs(data[["target","1"]], data[["2"]])

def test_check_inputs_first_residual_index_missing_raises():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types})
    residuals = data[["target"]].iloc[5:]
    variables = data[["1"]].iloc[10:]
    with pytest.raises(IndexError):
        asso._check_inputs(residuals, variables)

def test_check_inputs_last_variable_index_missing_raises():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types})
    residuals = data[["target"]].iloc[:10]
    variables = data[["1"]].iloc[:20]
    with pytest.raises(IndexError) as excinfo:
        asso._check_inputs(residuals, variables)
    # regression guard: the message must name the *last* index of variables_df (19),
    # not its first index (0), which was the pre-fix bug.
    assert str(variables.index[-1]) in str(excinfo.value)

def test_check_inputs_lags_exceeds_variables_length_raises():
    data, variable_types = _make_temporal_numerical_data()
    residuals = data[["target"]].iloc[:5]
    variables = data[["1"]].iloc[:5]
    asso = associations.PearsonMultivariate({"lags":10,"variable_types":variable_types})
    with pytest.raises(ValueError):
        asso._check_inputs(residuals, variables)

def test_check_inputs_check_na_residuals_raises():
    data, variable_types = _make_temporal_numerical_data()
    residuals = data[["target"]].iloc[:10].copy()
    residuals.iloc[5,0] = np.nan
    variables = data[["1"]].iloc[:10]
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types,"check_na":True})
    with pytest.raises(ValueError):
        asso._check_inputs(residuals, variables)

def test_check_inputs_check_na_variables_raises():
    data, variable_types = _make_temporal_numerical_data()
    residuals = data[["target"]].iloc[:10]
    variables = data[["1"]].iloc[:10].copy()
    variables.iloc[5,0] = np.nan
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types,"check_na":True})
    with pytest.raises(ValueError):
        asso._check_inputs(residuals, variables)

def test_check_inputs_check_na_false_by_default_allows_nan():
    data, variable_types = _make_temporal_numerical_data()
    residuals = data[["target"]].iloc[:10].copy()
    residuals.iloc[5,0] = np.nan
    variables = data[["1"]].iloc[:10]
    asso = associations.PearsonMultivariate({"lags":1,"variable_types":variable_types})
    asso._check_inputs(residuals, variables)  # must not raise

# D. _handle_constant_residuals

def test_handle_constant_residuals_returns_ones():
    data, variable_types = _make_temporal_numerical_data()
    asso = associations.PearsonMultivariate({"lags":3,"variable_types":variable_types})
    variables = data[["1","2"]].to_numpy()
    result = asso._handle_constant_residuals(variables)
    assert result.shape == (2,3)
    npt.assert_array_equal(result, np.ones((2,3)))

def test_handle_constant_residuals_shared_between_subclasses():
    data, variable_types = _make_temporal_mixed_data()
    config = {"lags":4,"categorical_method":"f_oneway","variable_types":variable_types}
    pearson_asso = associations.PearsonMultivariate(config)
    anova_asso = associations.ANOVATemporalSlow(config)
    variables = data[["1","2"]].to_numpy()
    npt.assert_array_equal(
        pearson_asso._handle_constant_residuals(variables),
        anova_asso._handle_constant_residuals(variables),
    )

# E. _select_correct_rows

def test_select_correct_rows_residuals_start_within_lag_window():
    # t3 < t1 + lags: variables index 0..19, residuals index 2..19, lags=5
    lags = 5
    variables_df = pd.DataFrame({"v": np.arange(20)}, index=range(20))
    residuals_df = pd.DataFrame({"r": 1000 + np.arange(2, 20)}, index=range(2, 20))
    asso = associations.PearsonMultivariate({"lags": lags, "variable_types": {"v": "numerical"}})
    residuals, variables = asso._select_correct_rows(residuals_df, variables_df)
    npt.assert_array_equal(residuals, 1000 + np.arange(5, 20))
    npt.assert_array_equal(variables.flatten(), np.arange(0, 20))

def test_select_correct_rows_residuals_start_after_lag_window():
    # t3 >= t1 + lags: variables index 0..19, residuals index 8..19, lags=5
    lags = 5
    variables_df = pd.DataFrame({"v": np.arange(20)}, index=range(20))
    residuals_df = pd.DataFrame({"r": 1000 + np.arange(8, 20)}, index=range(8, 20))
    asso = associations.PearsonMultivariate({"lags": lags, "variable_types": {"v": "numerical"}})
    residuals, variables = asso._select_correct_rows(residuals_df, variables_df)
    npt.assert_array_equal(residuals, 1000 + np.arange(8, 20))
    npt.assert_array_equal(variables.flatten(), np.arange(3, 20))

# F. _distribute_independence_tests

def test_distribute_independence_tests_matches_serial_apply():
    data, variable_types = _make_temporal_numerical_data()
    lags = 4
    residuals_df = data[["target"]]
    variables_df = data[["1","2","3","4"]]

    base_config = {"lags": lags, "variable_types": variable_types}
    reference_asso = associations.PearsonMultivariate({**base_config, "n_jobs": 1})
    residuals, variables = reference_asso._select_correct_rows(residuals_df, variables_df)
    expected = reference_asso._apply_independence_tests(residuals, variables)

    for n_jobs in (1, -1, 2):
        asso = associations.PearsonMultivariate({**base_config, "n_jobs": n_jobs})
        result = asso._distribute_independence_tests(residuals, variables)
        npt.assert_allclose(result, expected, atol=1e-8)