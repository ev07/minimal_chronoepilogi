from scipy.stats import pearsonr
import scipy.stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
import pingouin
import tigramite.independence_tests.regressionCI

import abc

import numpy as np
import pandas as pd

from chronoepilogi.util_mass_ts import mass2_modified
from chronoepilogi.associations import TemporalSlowAssociation, CrossSectionalAssociation

##########
#
#   Conditional Independence with Residuals
#
##########




class ResIndTest(abc.ABC):
    """
    Base class for approximate conditional independence tests of whether a
    candidate variable is redundant for explaining the residuals, once a
    condition variable is accounted for.

    Subclasses only need to implement `_check_config` (the config keys they
    require) and `ci_test` (the actual statistical test), including whatever
    data preparation their approach to the test needs.
    """
    def __init__(self, config):
        self.config = config
        self._check_config()

    @abc.abstractmethod
    def _check_config(self):
        """
        Verify that self.config contains the keys required by this test.
        To be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def ci_test(self, residuals_df, candidate_df, condition_df):
        """
        Compute the p-value(s) of the (partial) association between residuals
        and candidate, adjusting for condition. To be implemented by subclasses.
        """
        pass


class TemporalLinearRIT(ResIndTest):
    """
    Test whether the candidate is redundant given the condition, by fitting
    two nested regressions of residuals on lags of condition, and on lags of
    [condition, candidate], and running a likelihood-ratio test between them.

    Each regression is an ARMA-errors model (an ARIMA with exogenous
    regressors and an ARMA("lags","lags") error term) rather than plain OLS:
    residuals are a time series and are typically autocorrelated, and
    modeling that autocorrelation directly in the likelihood is what makes
    the classical likelihood-ratio (chi-squared) test valid here -- unlike an
    i.i.d.-errors OLS fit, whose likelihood would be distorted by the
    leftover autocorrelation.

    Note that ARIMA's exog regressors only enter the model contemporaneously
    (as `beta * exog_t`); the model does not itself look at past values of
    exog, that is only what its own ARMA error term does for the residuals.
    So to test whether any of the last "lags" values of condition/candidate
    help explain the residuals, `_prepare_data` builds one column per lag of
    each and passes all of them as exog columns, same as the OLS-based test
    used to. This is unrelated to, and does not conflict with, the model's
    own ARMA("lags","lags") error structure, which separately accounts for
    the autocorrelation of the residual series itself.
    """
    def _check_config(self):
        assert "lags" in self.config

    def _prepare_data(self, condition_df, residuals_df, candidate_df):
        """
        Format the lags of condition_df and candidate_df and join them with
        residuals_df in a single dataframe where rows are observation vectors.
        """
        # remove nans eventually occuring in residuals
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]

        # add lags of the condition variable
        col_name = condition_df.columns[0]
        condition_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            condition_cols[col_name+"lag -"+str(lag)] = condition_df[col_name].shift(lag)
        condition_cols = condition_cols.iloc[self.config["lags"]:]

        # add lags of the tested variable
        col_name = candidate_df.columns[0]
        candidate_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            candidate_cols[col_name+"lag -"+str(lag)] = candidate_df[col_name].shift(lag)
        candidate_cols = candidate_cols.iloc[self.config["lags"]:]

        # create new index
        new_index = residuals_df.index.intersection(condition_cols.index)
        residuals_df = residuals_df.loc[new_index]
        candidate_cols = candidate_cols.loc[new_index]
        condition_cols = condition_cols.loc[new_index]

        # concatenate
        df = pd.concat([residuals_df, candidate_cols, condition_cols],axis=1)
        cond_names = condition_cols.columns
        cand_names = candidate_cols.columns
        return df, cond_names, cand_names

    def ci_test(self, residuals_df, candidate_df, condition_df):
        residname = residuals_df.columns[0]
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)
        order = (self.config["lags"], 0, self.config["lags"])

        restricted_model = ARIMA(data[residname], exog=data[cond_names], order=order).fit()
        full_model = ARIMA(data[residname], exog=data[cond_names.tolist()+cand_names.tolist()], order=order).fit()

        lr_stat = -2 * (restricted_model.llf - full_model.llf)
        df_diff = len(cand_names)
        p_value = scipy.stats.chi2.sf(lr_stat, df_diff)
        return p_value

class CrossSectionalLinearRIT(ResIndTest):
    """
    Compute using two OLS models and a lr-test, whether the candidate is redundant given the condition.
    """
    def _check_config(self):
        assert "large_sample" in self.config

    def _prepare_data(self,condition_df, residuals_df, candidate_df):
        # just rename columns to avoid problems due to multiindex
        condition = condition_df.copy()
        condition.columns = ["cond"+str(i) for i in range(len(condition.columns))]
        candidate = candidate_df.copy()
        candidate.columns = ["cand"+str(i) for i in range(len(candidate.columns))]
        cond_names = condition.columns
        cand_names = candidate.columns
        return pd.concat([condition,candidate],axis=1), cond_names, cand_names


    def ci_test(self,residuals_df, candidate_df, condition_df):
        """
        :param residuals_df: pd.Series or pd.DataFrame
        :param candidate_df: pd.DataFrame
        :param condition_df: pd.DataFrame
        :return: np.array
        """
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)

        restricted_model = OLS(residuals_df, data[cond_names], missing="drop").fit()
        full_model = OLS(residuals_df, data[cond_names.tolist()+cand_names.tolist()], missing="drop").fit()
        lr_stat, p_value, df_diff = full_model.compare_lr_test(restricted_model, large_sample=self.config["large_sample"])
        return p_value




##########
#
#   Individual lags partial correlation test for the residuals.
#
##########




class LagPairsResidCITest(abc.ABC):
    """
    Base class for approximate conditional independence tests that work at
    the level of individual lag pairs: for each retained lag of candidate and
    each retained lag of condition, they estimate a partial-correlation
    p-value between residuals and that lag of candidate conditioning on that
    lag of condition (and symmetrically for condition given candidate), plus
    the unconditional p-value of each retained lag with residuals.

    Subclasses only need to implement `_check_config` (the config keys they
    require) and `partial_corr` (the actual per-lag-pair computation),
    returning (p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo).
    """
    def __init__(self, config):
        self.config = config
        self._check_config()

    @abc.abstractmethod
    def _check_config(self):
        """
        Verify that self.config contains the keys required by this test.
        To be implemented by subclasses.
        """
        pass

    def _select_correct_rows(self, residuals_df, variables_df):
        """
        Align residuals and variables for the correlation computation
        """
        # remove nans at begining of residuals
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        residuals_indexes = set(residuals_df.index)
        #adjust variable timestamps to residuals since learning process lags will have reduced the length of the series
        variables_ilocs = [i for i in range(variables_df.shape[0]) if (variables_df.index[i] in residuals_indexes)]
        #remove the first <lags> elements of the residuals for mass2_modified computation.
        residuals_ilocs = list(range(residuals_df.shape[0]))
        residuals_ilocs = residuals_ilocs[self.config["lags"]:]

        residuals = residuals_df.iloc[residuals_ilocs].values.reshape((-1,))
        variables = variables_df.iloc[variables_ilocs].values
        return residuals, variables

    def _correl_pvalue(self,r, n, k):
        """Compute the p-value of a partial correlation coefficient.
        https://pingouin-stats.org/build/html/_modules/pingouin/correlation.html#partial_corr
        """
        # using a student T distribution
        dof = n - k - 2
        tval = r * np.sqrt(dof / (1 - r**2 + 1e-16))  # in case 1-r**2==0
        pval = 2 * scipy.stats.t.sf(np.abs(tval), dof)
        return pval

    @abc.abstractmethod
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        """
        Compute the per-lag-pair p-value tables (p_RCa_Co, p_RCo_Ca) and the
        per-lag unconditional p-value vectors (p_RCa, p_RCo) between
        residuals and candidate/condition. To be implemented by subclasses.
        """
        pass


class HkPartialCorrelation(LagPairsResidCITest):

    def _check_config(self):
        assert "lags" in self.config
        assert "k" in self.config

    def partial_corr(self,residuals_df, candidate_df, condition_df):
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals

        # effective size over which the above has been estimated:
        sample_length = len(res)


        # correlation of Cand and Cond
        k = self.config["k"]
        Co_max_indexes = np.argpartition(np.abs(RCo),-k)[-k:] # only test the lag of Co with maximal correlation
        Ca_max_indexes = np.argpartition(np.abs(RCa),-k)[-k:]


        RCa_Co = np.zeros((k,k))
        RCo_Ca = np.zeros((k,k))
        # we align both tables for the correct lags.
        for i,Ca_max_index in enumerate(Ca_max_indexes):
            for j,Co_max_index in enumerate(Co_max_indexes):

                CoCa = pearsonr(cand[Ca_max_index:-self.config["lags"]+Ca_max_index][:,0],cond[Co_max_index:-self.config["lags"]+Co_max_index][:,0])[0]

                # if any division by 0 occur, this means that the condition is identical to either residuals or candidate.
                # so partial correlation is set to 0.
                a = RCa[Ca_max_index] - RCo[Co_max_index]*CoCa
                b = np.sqrt( (1 - CoCa**2)*(1 - RCo[Co_max_index]**2) )
                RCa_Co[i,j] = np.divide(a,b,where=b!=0,out=np.zeros_like(a))
                a = RCo[Co_max_index] - RCa[Ca_max_index]*CoCa
                b = np.sqrt( (1 - CoCa**2)*(1 - RCa[Ca_max_index]**2) )
                RCo_Ca[j,i] = np.divide(a,b,where=b!=0,out=np.zeros_like(a))


        # compute pvalue of the two correlation tables

        p_RCa_Co = self._correl_pvalue(RCa_Co,sample_length,1) # conditioning set is size 1
        p_RCo_Ca = self._correl_pvalue(RCo_Ca,sample_length,1) # conditioning set is size 1
        p_RCo = self._correl_pvalue(RCo[Co_max_indexes],sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa[Ca_max_indexes],sample_length,0) # conditioning set is size 0

        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo


class MixedTemporalSlowHk(LagPairsResidCITest):
    """Partial Correlation heuristic for categorical-categorical and continuous-categorical.
    """
    def _check_config(self):
        assert "variable_types" in self.config
        assert "k" in self.config
        assert "categorical_method" in self.config
        assert "lags" in self.config

    def partial_corr(self, residuals_df, candidate_df, condition_df):
        k = self.config["k"]
        lags = self.config["lags"]
        method = self.config["categorical_method"]
        variable_types = self.config["variable_types"]
        nameCa = candidate_df.columns[0]
        nameCo = condition_df.columns[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        #!TODO: replace by call to predefined class to avoid computing results twice
        asso_instance = TemporalSlowAssociation({"categorical_method":method,"lags":lags,"variable_types":variable_types})
        _ = asso_instance.association(residuals_df, pd.concat([candidate_df,condition_df],axis=1))

        p_RCa = np.array(asso_instance.pvalues[nameCa])
        p_RCo = np.array(asso_instance.pvalues[nameCo])
        Co_max_indexes = np.argpartition(-p_RCo, -k)[-k:]  # only test the lag of Co with maximal association
        Ca_max_indexes = np.argpartition(-p_RCa, -k)[-k:]

        p_RCa = p_RCa[Ca_max_indexes]
        p_RCo = p_RCo[Co_max_indexes]

        p_RCa_Co = np.zeros((k, k))
        p_RCo_Ca = np.zeros((k, k))

        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        res,cond=self._select_correct_rows(residuals_df, condition_df)

        #categorical-numerical and categorical-categorical
        if typeCa!="numerical" or typeCo!="numerical":
            n = len(res)
            x_type = np.zeros((n,1))
            y_type = np.zeros((n,1)) if typeCa!="categorical" else np.ones((n,1))
            z_type = np.zeros((n,1)) if typeCo!="categorical" else np.ones((n,1))

            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    instance = tigramite.independence_tests.regressionCI.RegressionCI()
                    x = np.expand_dims(res, axis=1)
                    y = cand[Ca_max_index:n+Ca_max_index]
                    z = cond[Co_max_index:n+Co_max_index]
                    p_RCa_Co[i,j] = instance.run_test_raw(x,y,z,x_type=x_type,y_type=y_type,z_type=z_type)[1]
                    p_RCo_Ca[j,i] = instance.run_test_raw(x,z,y,x_type=x_type,y_type=z_type,z_type=y_type)[1]

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo


class CrossSectionalHk(LagPairsResidCITest):
    """ Partial correlation for non-temporal, mixed type, grouped data, used during the equivalence phase.

    Notes
    -----
    This class is intended for use with two-level column index dataframes.
    The first level corresponds to groups of features, over which the association is computed.
    See documentation on data format for precisions.

    Given residuals denoted R, a candidate group Ca, a condition group Co,
    and noting feature i of group Ca by Ca_i and feature j of group Co by Co_j,
    This method computes:
        1) the pvalue of R indep Ca_i for all i
        2) the pvalue of R indep Co_j for all j
        3) the index i1,...,ik corresponding the maximal association between R and Ca_i (minimal pvalues)
        4) the index j1,...,jk corresponding the maximal association between R and Co_j (minimal pvalues)
        5) the pvalue of R indep Ca_iu | Co_jv, for iu in {i1,...,ik} and jv in {j1,...,jk}
        6) the pvalue of R indep Co_jv | Ca_iu, for iu in {i1,...,ik} and jv in {j1,...,jk}
    """
    def __init__(self, config:dict):
        """ Initialize the partial correlation object.

        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "variable_types": dict, for each group name (first level of the column index),
                whether it is "numerical" or "categorical".
                See examples.
             - "k": int, the number of features to consider for equivalence computation.
                If a group has lower than k features, all features are considered.
                k must be non-zero and positive.

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,7)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G1","c"),("G2","a"),("G2","b"),("G2","c")]))
        >>> variable_types = dict([(group, "numerical") for group in data.columns.get_level_values(0).unique()])
        >>> parcorr = CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
        >>> parcorr

        Or with mixed types:

        >>> numerical = pd.DataFrame(np.random.random(size=(1000,4)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b"),("G1","c")]))
        >>> categorical = pd.DataFrame(np.random.randint(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
        >>> parcorr =  CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
        >>> parcorr

        """
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "variable_types" in self.config
        assert "categorical_method" in self.config
        assert "k" in self.config

    def partial_corr(self,residuals_df: pd.DataFrame,
                     candidate_df: pd.DataFrame,
                     condition_df: pd.DataFrame) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Computes the partial correlations between features of two different groups.

        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (nsamples, 1) containing the model residuals of a learning model.
        candidate_df: pd.DataFrame
            DataFrame of shape (nsamples, groupsize1) containing the first group.
            The index must be aligned with residuals_df.
            The column index must have two levels, and with a unique group at level 0.
        condition_df: pd.DataFrame
            DataFrame of shape (nsamples, groupsize2) containing the second group.
            The index must be aligned with residuals_df.
            The column index must have two levels, and with a unique group at level 0.

        Returns
        -------
        p_RCa_Co: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Ca_i | Co_j),
            with R the residuals, Ca the candidate group, Co the condition group.
            The first dimension correspond to a retained feature of Ca, the second dimension to a feature of Co.
        p_RCo_Ca: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Co_j | Ca_i).
            The first dimension correspond to a retained feature of Co, the second dimension to a feature of Ca.
        p_RCa: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Ca_i).
        p_RCo: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Co_i).

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,7)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G1","c"),("G2","a"),("G2","b"),("G2","c")]))
        >>> variable_types = dict([(group, "numerical") for group in data.columns.get_level_values(0).unique()])
        >>> parcorr = CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
        >>> parcorr.partial_corr(data[["target"]], data[["G1"]], data[["G2"]])
        (array([[0.36888558, 0.33352269],[0.0014652 , 0.0013898 ]]),array([[0.2173927 , 0.25930918],[0.07060479, 0.08380666]]), array([0.3808198 , 0.00130095]), array([0.22332839, 0.07809404]))

        Or with mixed types:

        >>> rng = np.random.default_rng(0)
        >>> numerical = pd.DataFrame(rng.random(size=(1000,4)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b"),("G1","c")]))
        >>> categorical = pd.DataFrame(rng.integers(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
        >>> parcorr =  CrossSectionalHk({"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
        >>> parcorr.partial_corr(data[["target"]], data[["G1"]], data[["G2"]])
        (array([[0.59614113, 0.59506099],[0.03118282, 0.03196968]]),array([[0.73737096, 0.70131846],[0.60607526, 0.57785402]]),array([0.57745635, 0.03535778]), array([0.5959402 , 0.45854079]))
        """

        k = self.config["k"]
        variable_types = self.config["variable_types"]
        categorical_method = self.config["categorical_method"]
        nameCa = candidate_df.columns.get_level_values(0).unique()[0]
        nameCo = condition_df.columns.get_level_values(0).unique()[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        #!TODO: replace by call to predefined class to avoid computing results twice
        asso_config = {"variable_types":variable_types,"categorical_method":categorical_method}
        asso_instance = CrossSectionalAssociation(asso_config)
        _ = asso_instance.association(residuals_df, pd.concat([candidate_df,condition_df],axis=1))

        p_RCa = asso_instance.pvalues[nameCa]
        p_RCo = asso_instance.pvalues[nameCo]
        Co_max_indexes = np.argpartition(-p_RCo, -k)[-k:] if k<len(p_RCo) else np.argpartition(-p_RCo, -k)
        Ca_max_indexes = np.argpartition(-p_RCa, -k)[-k:] if k<len(p_RCa) else np.argpartition(-p_RCa, -k)

        p_RCa = p_RCa[Ca_max_indexes]
        p_RCo = p_RCo[Co_max_indexes]

        p_RCa_Co = np.zeros((k, k))
        p_RCo_Ca = np.zeros((k, k))

        #numerical-numerical
        if typeCa=="numerical" and typeCo=="numerical":
            # we align both tables for the correct lags.
            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    d = pd.concat([residuals_df,candidate_df[nameCa].iloc[:,Ca_max_index],condition_df[nameCo].iloc[:,Co_max_index]],axis=1)
                    d.columns = ["res","cand","cond"]
                    p_RCa_Co[i,j] = pingouin.partial_corr(data=d, x="res", y="cand", covar="cond")["p_val"].values[0]
                    p_RCo_Ca[j,i] = pingouin.partial_corr(data=d, x="res", y="cond", covar="cand")["p_val"].values[0]


        #categorical-numerical and categorical-categorical
        if typeCa!="numerical" or typeCo!="numerical":
            # define outside of loop to avoid allocating space repetitively
            n = len(candidate_df)
            x_type = np.zeros((n,1))
            y_type = np.zeros((n,1)) if typeCa!="categorical" else np.ones((n,1))
            z_type = np.zeros((n,1)) if typeCo!="categorical" else np.ones((n,1))
            # we align both tables for the correct lags.
            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    instance = tigramite.independence_tests.regressionCI.RegressionCI()
                    x = residuals_df
                    y = candidate_df[nameCa].iloc[:,[Ca_max_index]]
                    z = condition_df[nameCo].iloc[:,[Co_max_index]]
                    p_RCa_Co[i,j] = instance.run_test_raw(x,y,z,x_type=x_type,y_type=y_type,z_type=z_type)[1]
                    p_RCo_Ca[j,i] = instance.run_test_raw(x,z,y,x_type=x_type,y_type=z_type,z_type=y_type)[1]

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo



class TemporalSlowHk(LagPairsResidCITest):
    """ Partial correlation for mixed type data, during the equivalence phase.

    Notes
    -----
    Given residuals denoted R, a candidate variable Ca, a condition variable Co,
    and noting lag i of variable Ca by Ca_i and lag j of variable Co by Co_j,
    This method computes:
        1) the pvalue of R indep Ca_i for all i
        2) the pvalue of R indep Co_j for all j
        3) the index i1,...,ik corresponding the maximal association between R and Ca_i (minimal pvalues)
        4) the index j1,...,jk corresponding the maximal association between R and Co_j (minimal pvalues)
        5) the pvalue of R indep Ca_iu | Co_jv, for iu in {i1,...,ik} and jv in {j1,...,jk}
        6) the pvalue of R indep Co_jv | Ca_iu, for iu in {i1,...,ik} and jv in {j1,...,jk}
    """
    def __init__(self, config:dict):
        """ Initialize the partial correlation object.

        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "lags": int, the number of lags to compute the correlation over
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "variable_types": dict, for each variable name, whether it is "numerical" or "categorical".
                See examples.
             - "k": int, the number of lags to consider for equivalence computation.
               "k" must be lower or equal to "lags". k must be non-zero and positive.

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowHk({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"k":2})

        Or with mixed types:

        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowHk({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types,"k":2})
        """
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "variable_types" in self.config
        assert "k" in self.config
        assert "categorical_method" in self.config
        assert "lags" in self.config

    def partial_corr(self, residuals_df:pd.DataFrame,
                     candidate_df:pd.DataFrame,
                     condition_df:pd.DataFrame) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Computes the partial correlations between lags.

        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing the model residuals of a learning model.
        candidate_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing one of the two univariate time series to test for equivalence.
            The index must be aligned with residuals_df
        condition_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing one of the two univariate time series to test for equivalence.
            The index must be aligned with residuals_df

        Returns
        -------
        p_RCa_Co: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Ca_i | Co_j),
            with R the residuals, Ca the candidate TS, Co the condition TS.
            The first dimension correspond to a retained lag of Ca, the second dimension to a lag of Co.
        p_RCo_Ca: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Co_j | Ca_i).
            The first dimension correspond to a retained lag of Co, the second dimension to a lag of Ca.
        p_RCa: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Ca_i).
        p_RCo: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Co_i).

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.partial_corr(data[["target"]],data[["1"]],data[["2"]])
        (array([[0.07208953, 0.05627934],
                [0.03686298, 0.04137501]]),
        array([[0.09649547, 0.10936624],
                [0.02173326, 0.03464236]]),
        array([0.07455153, 0.03384917]),
        array([0.09990165, 0.02838155]))
        """

        variable_types = self.config["variable_types"]
        nameCa = candidate_df.columns[0]
        nameCo = condition_df.columns[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        if typeCa=="numerical" and typeCo=="numerical":
            partial_obj = HkPartialCorrelation(self.config)
            p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = partial_obj.partial_corr(residuals_df, candidate_df, condition_df)
        else:
            partial_obj = MixedTemporalSlowHk(self.config)
            p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = partial_obj.partial_corr(residuals_df, candidate_df, condition_df)

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo






# class TemporalSlowHk(CrossSectionalHk):
#     def reshape_to_cross_sectional(self,df):
#         lags = self.config["lags"]
#         data = dict()
#         for column in df.columns:
#             for l in range(lags):
#                 new_column = df[column].iloc[l:len(df)-lags+l]
#                 new_column.index = df.index[lags:]
#                 data[(column,l-lags)]=new_column
#         data = pd.DataFrame(data)
#         data.index = df.index[lags:]
#         data.columns = pd.MultiIndex.from_tuples(data.columns)
#         return data

#     def partial_corr(self,residuals_df, candidate_df, condition_df):
#         residuals_df = residuals_df
#         candidate_df = self.reshape_to_cross_sectional(candidate_df)
#         condition_df = self.reshape_to_cross_sectional(condition_df)
#         return super(TemporalSlowHk,self).partial_corr(residuals_df, candidate_df, condition_df)
