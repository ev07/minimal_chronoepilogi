import numpy as np
import pandas as pd
import pytest

from chronoepilogi import ChronoEpilogi
from chronoepilogi.models import OLSCrossSectional
from chronoepilogi.associations import TemporalSlowAssociation
from chronoepilogi.partial import TemporalSlowHk


#####################################################
#                                                   #
#     Coverage of ChronoEpilogi's init parameters   #
#                                                   #
#####################################################

# data providers

def _make_single_level_data(n=1000, ncols=12, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.random(size=(n,ncols)),columns=list(map(str,range(ncols))))
    data.loc[1:,"0"] = data["1"].shift(1) + data["2"].shift(1)
    data = data.fillna(0)
    return data

def _make_single_level_data_with_redundancy(n=1000, ncols=12, seed=0):
    data = _make_single_level_data(n=n, ncols=ncols, seed=seed)
    data["3"] = 0.4*data["1"]+0.3
    return data

def _make_two_level_data_with_signal(n=500, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.random(size=(n,7)),columns=pd.MultiIndex.from_tuples(
        [("0",""),("1","a"),("1","b"),("1","c"),("2","a"),("2","b"),("2","c")]))
    data[("0","")] = data[("1","a")] + data[("1","b")] + 0.1*rng.random(n)
    return data


# phases

@pytest.mark.parametrize("phases", ["F","FB","Fg","FgV","FBG","FBGV","FBE","FBEV"])
def test_chronoepilogi_fit_runs_for_all_phases(phases):
    data = _make_single_level_data()
    inst = ChronoEpilogi(data, "0", phases=phases)
    inst.fit()
    boundary = inst.get_first_markov_boundary()
    assert isinstance(boundary, list)
    assert set(boundary) == {"0","1","2"}


# equivalence_heuristic

@pytest.mark.parametrize("equivalence_heuristic", ["parcorr", "resid", "exact"])
def test_chronoepilogi_fit_runs_for_all_equivalence_heuristics(equivalence_heuristic):
    data = _make_single_level_data_with_redundancy()
    inst = ChronoEpilogi(data, "0", phases="FBEV", equivalence_heuristic=equivalence_heuristic)
    inst.fit()
    boundary = inst.get_first_markov_boundary()
    assert isinstance(boundary, list)
    assert set(boundary) == {"0","1","2"}
    classes = inst.get_equivalence_classes()
    assert isinstance(classes, list)
    # "1" and its near-duplicate "3" should end up recognized as equivalent
    assert any(set(c) == {"1","3"} for c in classes)


# start_with_univariate_autoregressive_model, for both single- and two-level data

@pytest.mark.parametrize("start_with_univariate_autoregressive_model", ["infer", True, False])
@pytest.mark.parametrize("data_format", ["single_level", "two_level"])
def test_chronoepilogi_fit_runs_for_all_autoregressive_settings(data_format, start_with_univariate_autoregressive_model):
    if data_format == "single_level":
        data = _make_single_level_data()
        target = "0"
    else:
        data = _make_two_level_data_with_signal()
        target = ("0","")

    if data_format == "two_level" and start_with_univariate_autoregressive_model is True:
        # documented as unsupported: "For double-level column index, should be
        # set to 'infer' or False, never True." _check_config rejects this
        # combination with a clear ValueError at construction time.
        with pytest.raises(ValueError):
            ChronoEpilogi(data, target, start_with_univariate_autoregressive_model=start_with_univariate_autoregressive_model)
        return

    inst = ChronoEpilogi(data, target, start_with_univariate_autoregressive_model=start_with_univariate_autoregressive_model)
    inst.fit()
    boundary = inst.get_first_markov_boundary()
    assert isinstance(boundary, list)
    if data_format == "single_level":
        if start_with_univariate_autoregressive_model in ("infer", True):
            assert set(boundary) == {"0","1","2"}
        else:
            assert set(boundary) == {"1","2"}
    else:
        assert "1" in boundary


# target_type

@pytest.mark.parametrize("target_type", ["continuous", "binary", "count"])
def test_chronoepilogi_fit_runs_for_all_target_types(target_type):
    rng = np.random.default_rng(0)
    if target_type == "continuous":
        data = _make_single_level_data()
    elif target_type == "binary":
        data = pd.DataFrame(rng.random(size=(300,8)),columns=list(map(str,range(8))))
        data["0"] = (data["1"] + rng.random(300) > 1).astype(int)
    else:  # count
        data = pd.DataFrame(rng.random(size=(300,8)),columns=list(map(str,range(8))))
        data["0"] = rng.poisson(lam=2, size=300)

    inst = ChronoEpilogi(data, "0", target_type=target_type)
    inst.fit()
    boundary = inst.get_first_markov_boundary()
    assert isinstance(boundary, list)
    assert "0" in boundary  # autoregressive by default on single-level data


# backward_removal_strategy

@pytest.mark.parametrize("backward_removal_strategy", ["first", "max"])
def test_chronoepilogi_fit_runs_for_all_backward_removal_strategies(backward_removal_strategy):
    data = _make_single_level_data()
    inst = ChronoEpilogi(data, "0", phases="FB", backward_removal_strategy=backward_removal_strategy)
    inst.fit()
    boundary = inst.get_first_markov_boundary()
    assert isinstance(boundary, list)
    assert set(boundary) == {"0","1","2"}



#####################################################
#                                                   #
#    Coverage of ChronoEpilogi's _check_config      #
#                                                   #
#####################################################

def test_check_config_invalid_phases_raises():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", phases="not-a-valid-phase")


def test_check_config_invalid_equivalence_early_stopping_raises():
    data = _make_single_level_data()
    with pytest.raises(TypeError):
        ChronoEpilogi(data, "0", equivalence_early_stopping="not-a-bool")


def test_check_config_invalid_target_type_raises():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", target_type="not-a-valid-target-type")


def test_check_config_invalid_equivalence_heuristic_raises():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", equivalence_heuristic="not-a-valid-heuristic")


def test_check_config_invalid_start_with_univariate_autoregressive_model_raises():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", start_with_univariate_autoregressive_model="not-valid")


def test_check_config_autoregressive_true_with_two_level_data_raises():
    data = _make_two_level_data_with_signal()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, ("0",""), start_with_univariate_autoregressive_model=True)


def test_check_config_invalid_backward_removal_strategy_raises():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", backward_removal_strategy="not-a-valid-strategy")


@pytest.mark.parametrize("given_kwargs", [
    {"model_class": OLSCrossSectional},
    {"model_config": {"constructor":{}, "fit":{}}},
])
def test_check_config_model_class_and_config_must_be_given_together(given_kwargs):
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", **given_kwargs)


@pytest.mark.parametrize("given_kwargs", [
    {"association_class": TemporalSlowAssociation},
    {"association_config": {"lags":1,"categorical_method":"f_oneway","variable_types":{}}},
])
def test_check_config_association_class_and_config_must_be_given_together(given_kwargs):
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", **given_kwargs)


@pytest.mark.parametrize("given_kwargs", [
    {"partial_correlation_class": TemporalSlowHk},
    {"partial_correlation_config": {"lags":1,"categorical_method":"f_oneway","variable_types":{},"k":1}},
])
def test_check_config_partial_correlation_class_and_config_must_be_given_together(given_kwargs):
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", **given_kwargs)


def test_check_config_model_test_method_required_with_explicit_model_class():
    data = _make_single_level_data()
    with pytest.raises(ValueError):
        ChronoEpilogi(data, "0", model_class=OLSCrossSectional,
                      model_config={"constructor":{}, "fit":{}, "residuals":"raw"})



#####################################################
#                                                   #
#        Coverage of ChronoEpilogi's updates        #
#                                                   #
#####################################################

# default_k -> partial_correlation_config["k"]

def test_default_k_update_affects_auto_partial_correlation_config():
    data = _make_single_level_data()
    inst = ChronoEpilogi(data, "0", phases="F", default_k=1)
    inst.fit()
    assert inst.partial_correlation_config["k"] == 1

    inst.fit(config={"default_k": 3})
    assert inst.partial_correlation_config["k"] == 3


def test_default_k_update_does_not_affect_explicit_partial_correlation_config():
    data = _make_single_level_data()
    partial_correlation_config = {"lags":1,"categorical_method":"f_oneway","variable_types":{},"k":5}
    inst = ChronoEpilogi(data, "0", phases="F",
                         partial_correlation_class=TemporalSlowHk,
                         partial_correlation_config=partial_correlation_config,
                         default_k=1)
    inst.fit()
    assert inst.partial_correlation_config["k"] == 5

    inst.fit(config={"default_k": 99})
    assert inst.partial_correlation_config["k"] == 5


# default_max_lag -> model_config / association_config / partial_correlation_config

def test_default_max_lag_update_affects_auto_configs_for_single_level_data():
    data = _make_single_level_data()
    inst = ChronoEpilogi(data, "0", phases="F", default_max_lag=1)
    inst.fit()
    assert inst.model_config["constructor"]["lags"] == 1
    assert inst.association_config["lags"] == 1
    assert inst.partial_correlation_config["lags"] == 1

    inst.fit(config={"default_max_lag": 4})
    assert inst.model_config["constructor"]["lags"] == 4
    assert inst.association_config["lags"] == 4
    assert inst.partial_correlation_config["lags"] == 4


@pytest.mark.parametrize("fixed_param", ["model", "association", "partial_correlation"])
def test_default_max_lag_update_does_not_affect_explicit_config(fixed_param):
    data = _make_single_level_data()
    variable_types = {column: "numerical" for column in data.columns}
    kwargs = {}
    if fixed_param == "model":
        kwargs["model_class"] = OLSCrossSectional
        kwargs["model_config"] = {"constructor":{}, "fit":{}, "residuals":"raw"}
        kwargs["model_test_method"] = "lr-test"
    elif fixed_param == "association":
        kwargs["association_class"] = TemporalSlowAssociation
        kwargs["association_config"] = {"lags":1,"categorical_method":"f_oneway","variable_types":variable_types}
    else:
        kwargs["partial_correlation_class"] = TemporalSlowHk
        kwargs["partial_correlation_config"] = {"lags":1,"categorical_method":"f_oneway","variable_types":variable_types,"k":1}

    inst = ChronoEpilogi(data, "0", phases="F", default_max_lag=1, **kwargs)
    inst.fit()

    fixed_configs_before = {
        "model": inst.model_config,
        "association": inst.association_config,
        "partial_correlation": inst.partial_correlation_config,
    }

    inst.fit(config={"default_max_lag": 6})

    fixed_configs_after = {
        "model": inst.model_config,
        "association": inst.association_config,
        "partial_correlation": inst.partial_correlation_config,
    }
    # the explicitly-given config is untouched by the default_max_lag update
    assert fixed_configs_after[fixed_param] == fixed_configs_before[fixed_param]

    # the remaining auto-inferred configs still pick up the new default_max_lag
    if fixed_param != "model":
        assert inst.model_config["constructor"]["lags"] == 6
    if fixed_param != "association":
        assert inst.association_config["lags"] == 6
    if fixed_param != "partial_correlation":
        assert inst.partial_correlation_config["lags"] == 6


def test_default_max_lag_update_does_not_affect_configs_for_two_level_data():
    data = _make_two_level_data_with_signal()
    inst = ChronoEpilogi(data, ("0",""), phases="F", default_max_lag=1)
    inst.fit()
    assert "lags" not in inst.model_config
    assert "lags" not in inst.association_config
    assert "lags" not in inst.partial_correlation_config

    inst.fit(config={"default_max_lag": 6})
    assert inst.default_max_lag == 6
    assert "lags" not in inst.model_config
    assert "lags" not in inst.association_config
    assert "lags" not in inst.partial_correlation_config