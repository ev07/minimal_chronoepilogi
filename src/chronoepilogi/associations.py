from scipy.stats import pearsonr,spearmanr, beta, rankdata, f_oneway, kruskal, alexandergovern
import scipy.stats

from joblib import Parallel, delayed, cpu_count

import numpy as np
import pandas as pd

from chronoepilogi.util_mass_ts import mass2_modified

##
#
#   Association classes
#
##


import abc
from typing import Any, Dict

class Association(abc.ABC):
    """
    Base class for association measures in ChronoEpilogi.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pvalues = dict()

    @abc.abstractmethod
    def association(self, residuals_df: pd.DataFrame, variable_df: pd.DataFrame) -> Any:
        """
        Computes the association measure between residuals and variables.
        To be implemented by subclasses.
        """
        pass





class LaggedAssociationBase(Association):
    """
    Shared machinery for lagged (temporal) independence tests between a single
    residual series and multiple candidate time series.

    Subclasses only need to implement `_apply_independence_tests` (the raw
    per-lag test statistic for each variable - a correlation coefficient for
    continuous data, or directly a p-value for a group-difference test such
    as ANOVA) and `association` (the orchestration returning -p-values).

    Notes
    -----
    The dataframe that contains the residuals must have only one column.
    The dataframe that contains the variables must have no missing data (nans) nor missing timestamps.
    The dataframe that contains the residuals can only have nan values before the first non-nan values, not after.
    The dataframe that contains the variables must begin at or before the first non-nan value of the residuals.
    The dataframe that contains the variables must end at or before the residuals end.
    The lags parameter of the configuration must be less than the length of the variables index.
    """

    def _cpus_from_njobs(self, n_jobs):
        """
        Gets the number of available cpu to configure the parallelism.
        """
        if n_jobs<0: return cpu_count()+1+n_jobs
        if n_jobs>0: return n_jobs
        raise ValueError("n_jobs cannot be equal to 0 (see joblib documentation).")

    def _remove_first_missings_from_residuals(self, residuals_df:pd.DataFrame):
        """
        Remove the timestamps with NaNs before the first non-NaN value.
        """
        first_index = residuals_df.iloc[:,0].first_valid_index()
        return residuals_df.loc[first_index:]

    def _check_inputs(self, residuals_df:pd.DataFrame, variables_df:pd.DataFrame):
        """
        Verifies the conditions in Notes.
        """
        check_na = self.config.get("check_na",False)
        lags = self.config["lags"]
        if len(residuals_df.columns)>1:
            raise ValueError("The residuals dataframe must have a single column. {} were provided.".format(len(residuals_df.columns)))
        if check_na:
            if residuals_df.isna().any(axis=None):
                raise ValueError("The residuals contain a NaN after the first non-nan value.")
            if variables_df.isna().any(axis=None):
                raise ValueError("The variables contain a NaN.")
        if residuals_df.index[0] not in variables_df.index:
            raise IndexError("The first index of residuals_df ({}) is not in variables_df.index.".format(residuals_df.index[0]))
        if variables_df.index[-1] not in residuals_df.index:
            raise IndexError("The last index of variables_df ({}) is not in residuals_df.index.".format(variables_df.index[-1]))
        if lags>=len(variables_df):
            raise ValueError("The lag parameter is {}, but the variables only have {} observations.".format(lags,len(variables_df)))
        if variables_df.index.get_loc(residuals_df.index[0]) < lags:
            if variables_df.index[lags] not in residuals_df.index:
                raise IndexError("Index {} should be in residuals_df.index, but is missing".format(variables_df.index[lags]))


    def _select_correct_rows(self, residuals_df:pd.DataFrame, variables_df:pd.DataFrame):
        """
        Adjust the residuals and variables indexes for mass computations.

        Notes
        -----
        If the variables span t1 to t2, and residuals from t3 to t4, we will select:
         - variables[t1:t2], residuals[t1+lags:t2] if t3<t1+lags
         - variables[t3-lags:t2], residuals[t3:t2] if t3>=t1+lags
        """
        lags = self.config["lags"]

        ## select residuals and variables so that variables starts lags step before resid.
        first_iloc = variables_df.index.get_loc(residuals_df.index[0])
        if first_iloc >= lags:
            variables_df = variables_df.iloc[first_iloc-lags:]
        else:
            first_iloc = residuals_df.index.get_loc(variables_df.index[lags])
            residuals_df = residuals_df.iloc[first_iloc:]

        ## select residuals so that variables ends at the same step as resid
        last_iloc = residuals_df.index.get_loc(variables_df.index[-1])
        residuals_df = residuals_df.iloc[:last_iloc+1]

        ## return numpy arrays
        residuals = residuals_df.to_numpy().reshape((-1,))
        variables = variables_df.to_numpy()
        return residuals, variables

    def _handle_constant_residuals(self, variables):
        """
        Return a p-value of 1 for every variable and lag: a constant residual
        cannot be associated with anything.
        """
        lags = self.config["lags"]
        return np.ones((variables.shape[1],lags))

    def _distribute_independence_tests(self, residuals, variables):
        """
        Handle the parallelization of the independence test computation according to the n_jobs parameter.
        """
        n_jobs = self.config.get("n_jobs", -1)
        cpus_available = self._cpus_from_njobs(n_jobs)
        column_split = np.array_split(list(range(variables.shape[1])),min(variables.shape[1],cpus_available))
        res = Parallel(n_jobs=n_jobs)(delayed(self._apply_independence_tests)(residuals, variables[:,list(cols)]) for cols in column_split)
        lagged_results = np.concatenate(list(res), axis=0)
        return lagged_results

    @abc.abstractmethod
    def _apply_independence_tests(self, residuals, variables):
        """
        Compute the raw per-lag test statistic for each variable: a correlation
        coefficient for continuous data, or directly a p-value for a
        group-difference test. To be implemented by subclasses.
        """
        pass


class PearsonMultivariate(LaggedAssociationBase):
    """
    Pearson Correlation for numerical data.
    """

    #!TODO have an option to directly get log-pvalues in case it is useful.
    #!TODO replace the pandas manipulations by a higher level data class. This will be hard though.

    def _is_pearsonr_faster(self, residuals_shape, variables_shape):
        """
        Chooses whether to use standard correlation or fft-accelerated correlation.
        """
        lags = self.config["lags"]
        #!TODO make a better choice of the fastest process between the fft and the pearsonr depending on T and lags.
        return lags == 1

    def _apply_independence_tests(self, residuals, variables):
        """
        Compute the correlations without using fft acceleration.
        """
        lags = self.config.get("lags", 1)
        lagged_correlations = [[pearsonr(residuals,variables[lags-l:-l,i]).correlation\
                        if not len(np.unique(variables[lags-l:-l,i]))==1 else 0\
                        for l in range(lags, 0,-1)] \
                        for i in range(variables.shape[1])]  # constant data verification is done here.
        lagged_correlations = np.array(lagged_correlations)
        return lagged_correlations


    def _apply_mass2(self, residuals, variables):
        """
        Parallelizes the computation of the correlation with fft acceleration.
        """
        n_jobs = self.config.get("n_jobs", -1)
        cpus_available = self._cpus_from_njobs(n_jobs)
        column_split = np.array_split(list(range(variables.shape[1])),min(variables.shape[1],cpus_available))
        res = Parallel(n_jobs=n_jobs)(delayed(mass2_modified)(variables[:,list(cols)], residuals) for cols in column_split)
        coefficients = np.concatenate(list(res), axis=0)
        return coefficients

    def _to_pvalues(self, lagged_correlations, sample_size:int):
        """
        For the pearson r coefficient, compute the p-value using the beta distribution.
        """
        # next 3 lines taken from scipy.stats.pearsonr
        ab = sample_size/2 - 1 
        beta_distribution = beta(ab, ab, loc=-1, scale=2)
        pvalues = 2 * beta_distribution.sf(np.abs(lagged_correlations))
        return pvalues
    
    def _to_logpvalues(self, lagged_correlations, sample_size:int):
        """
        For the pearson r coefficient, compute the log-p-value using the beta distribution.
        """
        # next 3 lines taken from scipy.stats.pearsonr
        ab = sample_size/2 - 1 
        beta_distribution = beta(ab, ab, loc=-1, scale=2)
        pvalues = 2 * beta_distribution.logsf(np.abs(lagged_correlations))
        return pvalues
    
    def _compute_ranks(self,residuals,variables):
        rr = rankdata(residuals)
        rv = rankdata(variables,axis=0)
        return rr,rv
    
    def _to_pvalues_spearman(self,lagged_coefficients, sample_size):
        """
        For the spearman r coefficient, compute the p-value using the student distribution.
        """
        # next lines taken from scipy.stats
        dof = sample_size - 2
        coefficients = lagged_coefficients * np.sqrt((dof/((lagged_coefficients+1.0)*(1.0-lagged_coefficients))).clip(0))
        coefficients = scipy.stats.t.sf(np.abs(coefficients), dof)*2
        return coefficients

    def _to_logpvalues_spearman(self,lagged_coefficients, sample_size):
        """
        For the spearman r coefficient, compute the log-p-value using the student distribution.
        """
        # next lines taken from scipy.stats
        dof = sample_size - 2
        coefficients = lagged_coefficients * np.sqrt((dof/((lagged_coefficients+1.0)*(1.0-lagged_coefficients))).clip(0))
        coefficients = scipy.stats.t.logsf(np.abs(coefficients), dof)*2
        return coefficients
    

    def association(self, residuals_df, variables_df):
        numerical_method = self.config.get("numerical_method","pearsonr")
        if numerical_method not in ("pearsonr", "spearmanr"):
            raise ValueError(f"Unknown numerical_method: {numerical_method}. Must be one of 'pearsonr' and 'spearmanr'.")

        residuals_df = self._remove_first_missings_from_residuals(residuals_df)
        self._check_inputs(residuals_df, variables_df)
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        if numerical_method == "spearmanr":
            residuals, variables = self._compute_ranks(residuals,variables)

        # constant residuals: no variable can be associated with them.
        if len(np.unique(residuals))==1:
            pvalues = self._handle_constant_residuals(variables)
        else:
            # pearsonr would be faster as there are too few lags for the fft to be worth it
            if self._is_pearsonr_faster(residuals.shape, variables.shape):
                lagged_correlations = self._distribute_independence_tests(residuals, variables)
            else:
                lagged_correlations = self._apply_mass2(residuals, variables)
            # transform to p-values
            if numerical_method == "spearmanr":
                pvalues = self._to_pvalues_spearman(lagged_correlations, residuals.shape[0])
            else:
                pvalues = self._to_pvalues(lagged_correlations, residuals.shape[0])

        self.pvalues = dict((variable, pvalues[i])for i,variable in enumerate(variables_df.columns))
        return np.max(-pvalues, axis=-1)


class ANOVATemporalSlow(LaggedAssociationBase):
    """
    Independence test for numerical residuals and categorical variables.
    """
    #!TODO vectorize as much as possible _apply_independence_tests

    def _apply_independence_tests(self, residuals, variables):
        lags = self.config.get("lags", 1)
        categorical_method = self.config["categorical_method"]

        lagged_pvalues = np.zeros((variables.shape[1],lags))
        for variable in range(variables.shape[1]):
            ncategories = sorted(np.unique(variables[:,variable]))
            categorical_filters = [variables[:,variable]==category for category in ncategories]

            for l in range(lags):
                samples = [residuals[mask[l:-lags+l]] for mask in categorical_filters]
                # we must avoid length zero groups of values
                samples = [s for s in samples if len(s)>0]
                
                if categorical_method == "f_oneway":
                    pval = f_oneway(*samples).pvalue if len(samples)>1 else 1.  # if one sample only, no link.
                elif categorical_method == "kruskal":
                    pval = kruskal(*samples).pvalue if len(samples)>1 else 1.
                elif categorical_method == "alexandergovern":
                    pval = alexandergovern(*samples).pvalue if len(samples)>1 else 1.
                else:
                    raise ValueError("Configuration categorical_method must be one of f_oneway, kruskal, alexandergovern.")

                lagged_pvalues[variable,l] = pval
        return lagged_pvalues

    def association(self, residuals_df, variables_df):
        residuals_df = self._remove_first_missings_from_residuals(residuals_df)
        self._check_inputs(residuals_df, variables_df)
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        # constant residuals: no variable can be associated with them.
        if len(np.unique(residuals))==1:
            pvalues = self._handle_constant_residuals(variables)
        else:
            pvalues = self._distribute_independence_tests(residuals, variables)
        self.pvalues = dict((variable, pvalues[i])for i,variable in enumerate(variables_df.columns))

        return np.max(-pvalues, axis=-1)



class TemporalSlowAssociation(Association):
    """Temporal data mixed-type association.

    Notes
    -----
    For continuous data, we use Pearson Correlation with mass implementation.
    For categorical data, we use an ANOVA test between the residuals and the tested series.
    """

    def __init__(self, config: dict):
        """ 
        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "lags": int, the number of lags to compute the correlation over
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "numerical_method": str, any of 'pearsonr', 'spearmanr'.
             - "variable_types": dict, for each variable name, whether it is "numerical" or "categorical".
                See examples.
             - "n_jobs": int, the number of processors used in parallel. Must be different from 0. See joblib.Parallel for more information.
             - "check_na": bool, if True, checks that there is no NaN in the variables and residuals DataFrames.

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso

        Or with mixed types:

        >>> numerical = pd.DataFrame(np.random.random(size=(1000,3)),columns=["target","1","2"])
        >>> categorical = pd.DataFrame(np.random.randint(size=(1000,2)),columns=["3","4"])
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso
        """
        super().__init__(config)


    def association(self,residuals_df:pd.DataFrame, variables_df:pd.DataFrame)-> np.array:
        """ 
        Computes the association score between the residuals and candidate time series.

        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing the model residuals of a learning model. 
            The index must be aligned with variables_df.
        variables_df: pd.DataFrame
            DataFrame of shape (ntimesteps, D) containing the D time series to test for association with the residuals.
            The index must be aligned with residuals_df

        Returns
        -------
        pvalues: np.array
            A 1D numpy array containing minus the minimal p-value across lags, for each of the D time series to test.
            The coefficients are in the same order as the columns in variables_df.columns.
            We return minus the p-value by convention, as the maximal -pvalue correspond to the maximal association.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]], data[["1","2","3","4"]])
        array([-0.03384917, -0.02838155, -0.0633841 , -0.15107386])

        Or with mixed types:

        >>> rng = np.random.default_rng(0)
        >>> numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
        >>> categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]], data[["1","2","3","4"]])
        array([-0.03111284, -0.04568282, -0.03302831, -0.02551908])
        """

        variable_types = self.config["variable_types"]

        # numerical TS
        pearson_obj = PearsonMultivariate(self.config)
        numerical_variables = [x for x in variables_df.columns if variable_types[x]=="numerical"]
        if len(numerical_variables)>0:
            numerical_pvalues = pearson_obj.association(residuals_df, variables_df[numerical_variables])
        else:
            numerical_pvalues = []

        # categorical TS
        anova_obj = ANOVATemporalSlow(self.config)
        categorical_variables = [x for x in variables_df.columns if variable_types[x]=="categorical"]
        if len(categorical_variables)>0:
            categorical_pvalues = anova_obj.association(residuals_df, variables_df[categorical_variables])
        else:
            categorical_pvalues = []

        # mix
        index_num, index_cat = 0,0
        pvalues = []
        for variable in variables_df.columns:
            if variable_types[variable] == "numerical":
                pvalues.append(numerical_pvalues[index_num]) 
                index_num+=1
            elif variable_types[variable] == "categorical":
                pvalues.append(categorical_pvalues[index_cat])
                index_cat+=1
            else:
                raise ValueError(f"Unknown variable type '{variable_types[variable]}' for variable '{variable}'. Must be 'numerical' or 'categorical'.")

        self.pvalues = {**pearson_obj.pvalues, **anova_obj.pvalues}
        return np.array(pvalues)


class CrossSectionalAssociation(Association):
    """Cross-sectional, mixed-type, grouped data association.

    Notes
    -----
    This class is intended for use with two-level column index dataframes.
    The first level corresponds to groups of features, over which the association is computed.
    See documentation on data format for precisions.

    For continuous data, we use Pearson Correlation.
    For categorical data, we use an ANOVA test between the residuals and the tested series.
    """
    def __init__(self, config: dict):
        """ 
        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "numerical_method": str, any of 'pearsonr', 'spearmanr'.
                This specifies the kind of test used for numerical data.
             - "variable_types": dict, for each group name (first level of the column index),
                whether it is "numerical" or "categorical". 
                This implies that all columns in a group must belong to the same type (numerical or categorical).
                See examples.
             - "n_jobs": int, the number of jobs for parallelism. See joblib.Parallel for details.

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,5)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G2","a"),("G2","b")]))
        >>> variable_types = dict([(group, "numerical") for group in data.columns.get_level_values(0).unique()])
        >>> asso = CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso

        Or with mixed types:

        >>> numerical = pd.DataFrame(np.random.random(size=(1000,3)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b")]))
        >>> categorical = pd.DataFrame(np.random.randint(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
        >>> asso =  CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso
        """
        super().__init__(config)

    def association(self, residuals_df: pd.DataFrame, variables_df: pd.DataFrame) -> np.ndarray:
        """
        Computes the association score between the residuals and candidate time series.

        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (nsamples, 1) containing the model residuals of a learning model. 
            The index must be aligned with variables_df.
        variables_df: pd.DataFrame
            DataFrame of shape (nsamples, D) containing the D features to test for association with the residuals.
            The index must be aligned with residuals_df.
            The columns must be a pd.MultiIndex instance with two levels.
            See documentation on data format for precisions.

        Returns
        -------
        pvalues: np.array
            A 1D numpy array containing minus the minimal p-value for each group defined by the first level column index.
            The coefficients are in the same order as the first level of the column index.
            We return minus the p-value by convention, as the maximal -pvalue correspond to the maximal association.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,5)),columns=pd.MultiIndex.from_tuples([("target",""),("G1","a"),("G1","b"),("G2","a"),("G2","b")]))
        >>> variable_types = dict([(column, "numerical") for column in data.columns.get_level_values(0).unique()])
        >>> asso = CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]], data[["G1","G2"]])
        array([-0.32736175, -0.11320393])

        Or with mixed types:

        >>> rng = np.random.default_rng(0)
        >>> numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=pd.MultiIndex.from_tuples([("target",None),("G1","a"),("G1","b")]))
        >>> categorical = pd.DataFrame(rng.integers(0,5,size=(1000,3)),columns=pd.MultiIndex.from_tuples([("G2","a"),("G2","b"),("G2","c")]))
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","G1":"numerical","G2":"categorical"}
        >>> asso =  CrossSectionalAssociation({"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]],data[["G1","G2"]])
        array([-0.05543262, -0.0992026 ])
        """
        variable_types = self.config["variable_types"]
        col_names = variables_df.columns.get_level_values(0).unique()
        n_jobs = self.config.get("n_jobs", -1)

        if len(np.unique(residuals_df.values.flatten()))==1:
            group_pvalues = np.ones((len(col_names)),dtype=np.float16)
            self.pvalues = {var:np.ones((len(variables_df[var].columns))) for var in col_names}
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(self._compute_group_association)(residuals_df, variables_df[variable], variable_types[variable]) for variable in col_names)
            group_pvalues = np.array([np.min(pvals) for pvals in results])
            self.pvalues = {var:res for var,res in zip(col_names, results)}
        return -group_pvalues
    
    def _compute_group_association(self, residuals_df: pd.DataFrame, group_df: pd.DataFrame, var_type: str) -> np.ndarray:
        if var_type == "numerical":
            pvals = self._numerical_group_pvalues(residuals_df, group_df)
        elif var_type == "categorical":
            pvals = self._categorical_group_pvalues(residuals_df, group_df)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
        return pvals

    def _numerical_group_pvalues(self, residuals_df, group_df):
        method = self.config.get("numerical_method","pearsonr")
        pval = np.zeros((len(group_df.columns),))
        for i,element in enumerate(group_df.columns):
            if len(np.unique(group_df[element].values.flatten()))==1:
                pval[i] = 1.
            elif method == "pearsonr":
                result = pearsonr(group_df[element], residuals_df[residuals_df.columns[0]])
                pval[i] = result.pvalue if hasattr(result, 'pvalue') else result[1]
            elif method == "spearmanr":
                result = spearmanr(group_df[element], residuals_df[residuals_df.columns[0]])
                pval[i] = result.pvalue if hasattr(result, 'pvalue') else result[1]
            else:
                raise ValueError(f"Unknown numerical_method: {method}. Must be one of 'pearsonr' or 'spearmanr'.")

        return pval

    def _categorical_group_pvalues(self, residuals_df, group_df):
        method = self.config["categorical_method"]
        pval = np.zeros((len(group_df.columns),))
        for i,element in enumerate(group_df.columns):
            ncategories = sorted(group_df[element].unique())
            samples = [residuals_df[residuals_df.columns[0]][group_df[element] == k] for k in ncategories]
            
            if len(ncategories) == 1:
                pval[i] = 1.
            elif method == "f_oneway":
                res = f_oneway(*samples)
                pval[i] = res.pvalue if hasattr(res,'pvalue') else res[1]
            elif method == "kruskal":
                res = kruskal(*samples)
                pval[i] = res.pvalue if hasattr(res,'pvalue') else res[1]
            elif method == "alexandergovern":
                res = alexandergovern(*samples)
                pval[i] = res.pvalue if hasattr(res,'pvalue') else res[1]
            else:
                raise ValueError(f"Unknown categorical_method: {method}")

        return pval
