import numpy as np
import pandas as pd
import pytest

from chronoepilogi import ChronoEpilogi


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
