import numpy as np
import numpy.testing as npt
import pandas as pd

from chronoepilogi import ChronoEpilogi


#####################################################
#                                                   #
#                Human-generated tests              #
#                                                   #
#####################################################

def test_chronoepilogi_first_solution():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(10001,100)),columns=list(map(str,range(100))))
    data.loc[1:,"0"] = data["1"].shift(1) + data["2"].shift(1)
    tss_instance = ChronoEpilogi(data, "0")
    tss_instance.fit()
    boundary = tss_instance.get_first_markov_boundary()
    assert set(boundary) == set(['0', '1', '2'])

def test_chronoepilogi_number_mb():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(10001,100)),columns=list(map(str,range(100))))
    data.loc[1:,"0"] = data["1"].shift(1) + data["2"].shift(1)
    data["3"] = 0.4*data["1"]+0.3
    tss_instance = ChronoEpilogi(data, "0", phases="FBEV")
    tss_instance.fit()
    number = tss_instance.get_total_number_markov_boundaries()
    assert number == 2

def test_chronoepilogi_from_index():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(10001,100)),columns=list(map(str,range(100))))
    data.loc[1:,"0"] = data["1"].shift(1) + data["2"].shift(1)
    data["3"] = 0.4*data["1"]+0.3
    tss_instance = ChronoEpilogi(data, "0", phases="FBEV")
    tss_instance.fit()
    mb1, mb2 = tss_instance.get_markov_boundary_from_index(0), tss_instance.get_markov_boundary_from_index(1)
    assert set(mb1) == set(['0', '2', '1'])
    assert set(mb2) == set(['0', '2', '3'])

def test_chronoepilogi_classes():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random(size=(10001,100)),columns=list(map(str,range(100))))
    data.loc[1:,"0"] = data["1"].shift(1) + data["2"].shift(1)
    data["3"] = 0.4*data["1"]+0.3
    tss_instance = ChronoEpilogi(data, "0", phases="FBEV")
    tss_instance.fit()
    classes = tss_instance.get_equivalence_classes()

    assert len(classes) == 3
    assert set(classes[0]) in [{"0"},{"2"},{"1","3"}]
    assert set(classes[1]) in [{"0"},{"2"},{"1","3"}]
    assert set(classes[2]) in [{"0"},{"2"},{"1","3"}]
    
    assert {"0"} in list(map(set,classes))
    assert {"2"} in list(map(set,classes))
    assert {"1","3"} in list(map(set,classes))

