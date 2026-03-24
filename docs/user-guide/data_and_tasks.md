# Data formats and predictive tasks that ChronoEpilogi handles

ChronoEpilogi was primarily created to solve Time Series Selection for the one-step-ahead univariate point forecasting task.
However, the algorithm has been extended to several other settings, mainly through different data format and the use of autoregressive models.

## Tasks

We will define the scope of ChronoEpilogi.

Predictive tasks vocabulary:

- *Supervised vs unsupervised*: Supervised learning tasks require the knowledge of the predicted target as part of the training procedure. Unsupervised analyse patterns in the input data and produce a prediction according to the observed patterns.
- *Numerical vs categorical target*: Different learning models are typically required depending on whether the predicted target is numerical (a quantity) or categorical (an unordered attribute).
- *Forecasting vs regression vs classification*: We generally refer to forecasting when the observations used in the tasks are not identically and independently distributed (iid) but correlated along time. Usually, the learning dataset is extracted from a single MTS using a sliding window. We refer to regression/classification when each observation is iid, in which case the dataset consists in a collection of MTS corresponding to independent sources (users, locations...). We note that *regression* is typically used for numerical predictions and *classification* for categorical predictions in the iid case, while *forecasting* is used whether the target is numerical or categorical.
- *Single-horizon forecasting vs multi-horizon forecasting*: In forecasting tasks, single horizon forecasting predict a single timestamp. It can be the timestamp $t$ immediately after the input $X_{t-L:t-1}$ (one-step-ahead prediction), or several timestamps ahead $t+h$. Multihorizon forecasting predicts several quantities at the same time.
- *Autoregressive vs non-autoregressive forecasting*: In autoregressive forecasting, the past of the predicted quantity is observed.

Selection tasks:

- *Time Series Selection*: in a MTS, select entire Time Series as predictors, regardless of the specific temporal dependencies.
- *Feature Selection*: in 1D data, select attributes/features as predictors individually.
- *Group Feature Selection*: in 1D data where attributes/features are separated into groups, select entire groups of attributes as predictors, regardless of the specific in-group dependencies.

ChronoEpilogi handles the above three Selection tasks. The Selection task is determined from the data format given to the algorithm.
ChronoEpilogi does not handle unsupervised tasks, as it requires target knowledge to find a Markov Boundary. ChronoEpilogi handles both forecasting and regressions tasks. The task is determined from the data format. 

Finally, the current implementation of ChronoEpilogi handles numerical-like predictions only: continuous, count and binary. As we rely on modeling error to decide on the next selected TS/group, we require that a single number represents entirely the error made on a specific prediction. Such residuals can be defined for continuous, count and binary predictions.

For the same reason of requiring a single number to completely represent the error made by the model, we only handle single horizon forecasting. Predicting several quantities would requires several error terms, which is beyond the current implementation.

## Data format

Data is provided to the algorithm in the form of a pandas DataFrame. 

For Time Series Selection, we use 1-level column indexes:

- For MTS data with one-step-ahead predictions, each row must correspond to a timestamp, and timestamps must be ordered at a regular frequency without missing timestamps. Each column of the DataFrame represents a TS. If the task is autoregressive, set ChronoEpilogi kwarg `start_with_univariate_autoregressive_model` to True, otherwise set it to False.
- For MTS data where the prediction's past is unobserved, the one-step-ahead prediction framework can be used, by setting the prediction $y$ corresponding to an observed window $X_{t-L:t-1}$ at timestamp $t$. Set ChronoEpilogi kwarg  `start_with_univariate_autoregressive_model` to False to avoid using previous predictions as part of the input.
- For MTS data with a prediction several timesteps ahead, create a new column in the DataFrame recording the prediction at time $t$, and set `start_with_univariate_autoregressive_model` to False.

For (Group) Feature Selection, we use 2-levels column indexes:

- Each row of the dataframe must correspond to one observation. If the input data is originally 2D, it must be flattened.
- The predicted target must be on the same row as the corresponding observation.
- For Group Feature Selection, the columns should be indexed by a pandas MultiIndex class with two levels. The first level correspond to a group name, the second level corresponds to the name of the feature/attribute.
- For Feature Selection, set group names identically to feature/attributes, ensuring a single attribute per group.
- Set `start_with_univariate_autoregressive_model` to `False` or `"infer"`.

The algorithm does not handle missing values. All columns and levels must be named with strings.

## Examples

The UCI machine learning repository provides datasets that suit a variety of tasks. We will use the Appliance Energy Prediction dataset to illustrate the above scenarios.

``` py
from chronoepilogi import ChronoEpilogi
from ucimlrepo import fetch_ucirepo

# fetch dataset 
appliances_energy_prediction = fetch_ucirepo(id=374) 
# data (as pandas dataframes) 
X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets
# merge
data = pd.merge(X,y,left_index=True, right_index=True)
# remove date
data = data[data.columns[1:]]
```

The Appliances Energy Prediction dataset ([Paper](https://doi.org/10.1016/j.enbuild.2017.01.083)) is a regression dataset for energy consumption prediction in a home. It has 28 TS and 19735 observations, and all TS record numerical quantities. Predictors include Temperature and Humidity records in several rooms, light fixture energy consumption, and outside meteorological indicators of temperature, humidity, pressure, wind speed, visibility and dewpoint.

The dataset was originally used in a non-temporal context, where the relation of TS was only examined contemporaneously. The authors wanted to identify models and predictors to solve the task of modeling Applicances at time $t$ given the other TS at time $t$. We will show other ways of setting the problem. For convenience, we note Appliances at time $t$ by $X^1_t$ and other TS at time $t$ by $X^j_t$, and denote their past by $X^j_{t-L:t-1}$.

### One-step-ahead prediction on MTS data

ChronoEpilogi with default parameters allow us to model Appliances at time $t$ in function of the past $L=10$ observations (100 minutes).
Formally, we model $X^1_t = f(X^{\boldsymbol{S}}_{t-L:t-1})$ with $\boldsymbol{S}$ a subset of the available TS, including Applicances past $X^1_{t-L:t-1}$.

``` py
variable_types = {c:"numerical" for c in data.columns}
target_type = "continuous"
target = "Appliances"
default_max_lags = 10

fs_instance = ChronoEpilogi(data, target, target_type=target_type, default_max_lag=default_max_lags, variable_types=variable_types)
fs_instance.fit()
fs_instance.get_first_markov_boundary()
```

```
['Appliances', 'lights', 'RH_5', 'T6', 'T1', 'RH_1', 'RH_9', 'Press_mm_hg']
```


### One-step-ahead prediction on MTS data without including the past of the forecasted series

We might not have the energy consumption of the applicances available at prediction time. Hence, we exclude the past of Appliances.


``` py hl_lines="7"
variable_types = {c:"numerical" for c in data.columns}
target_type = "continuous"
target = "Appliances"
default_max_lags = 10

fs_instance = ChronoEpilogi(data, target, target_type=target_type, default_max_lag=default_max_lags, variable_types=variable_types,
                            start_with_univariate_autoregressive_model=False)
fs_instance.fit()
fs_instance.get_first_markov_boundary()
```

```
['lights','RH_out','RH_1','RH_8','T4','T3','RH_2','T2','Windspeed','T8','Visibility','RH_9','RH_5','T9']
```

### Five-steps-ahead prediction on MTS data

To go from one-step-ahead to another horizon, we shift the forecasted series backwards. For the past of the forecasted quantity to be included in the model, we create another time series "Applicances_past" that will serve as predictor, and exclude the past of the target series "Appliances".

``` py hl_lines="1 2 3 4"
data2 = data.copy()
data2 = data2.rename(columns={"Appliances":"Appliances_past"})
data2["Appliances"] = data2["Appliances_past"].shift(-4)  # Put time t+4 at time t
data2 = data2.dropna()
variable_types = {c:"numerical" for c in data2.columns}
target_type = "continuous"
target = "Appliances"
default_max_lags = 10

fs_instance = ChronoEpilogi(data2, target, target_type=target_type, default_max_lag=default_max_lags, variable_types=variable_types,
                            start_with_univariate_autoregressive_model=False)
fs_instance.fit()
fs_instance.get_first_markov_boundary()
```

```
['Appliances_past','RH_5','RH_9','RH_2','lights','T1','T6','Press_mm_hg','T8','RH_3','RH_out']
```

### Feature Selection (atemporal)

We can use ChronoEpilogi to operate Feature Selection corresponding to the dataset original paper task.
Formally, we model $X^1_t = f(X^{\boldsymbol{S}}_{t})$ with $\boldsymbol{S}$ a subset of the available TS, including Applicances past $X^1$.
To do so, we use a MultiIndex column to indicate non-temporal data to the algorithm. As a result, `"variable_types"` become group types corresponding to the first level of the column index. Target become the name of the predicted quantity at both levels.


``` py hl_lines="2 3 5"
data3 = data.copy()
data3.columns = pd.MultiIndex.from_tuples([(c,c) for c in data3.columns])
variable_types = {c[0]:"numerical" for c in data3.columns}  # Group Type
target_type = "continuous"
target = ("Appliances","Appliances")

fs_instance = ChronoEpilogi(data3, target, target_type=target_type, variable_types=variable_types)
fs_instance.fit()
fs_instance.get_first_markov_boundary()
```

```
['lights','RH_out','RH_1','RH_8','T9','T3','RH_2','Windspeed','T2','T8','T6','Visibility','RH_3','RH_7','RH_5','T4','T_out','RH_6']
```

### Group Feature selection (atemporal)

The previous example specified that each feature was alone in its group, corresponding to the typical Feature Selection setting. We can do Group Feature Selection by specifying a group in the first level of the column index. In the following example, Temperature and Humidity corresponding to a same room in the house were grouped together.

``` py hl_lines="2 3"
data4 = data.copy()
group = lambda c: "out" if c=="T_out" or c=="RH_out" else (c[-1] if len(c)==2 or c[:3]=="RH_" else c)
data4.columns = pd.MultiIndex.from_tuples([(group(c),c) for c in data4.columns])
variable_types = {c[0]:"numerical" for c in data4.columns}
target_type = "continuous"
target = ("Appliances","Appliances")

fs_instance = ChronoEpilogi(data4, target, target_type=target_type, variable_types=variable_types)
fs_instance.fit()
fs_instance.get_first_markov_boundary()
```

```
['lights', 'out', '1', '8', '2', '3', 'Windspeed', '9', 'Visibility', '6']
```