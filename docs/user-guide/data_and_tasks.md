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
- Set `start_with_univariate_autoregressive_model` to False.

The algorithm does not handle missing values. All columns and levels must be named with strings.

## Examples

The UCI machine learning repository provides datasets that suit a variety of tasks. We will use several of the availabel datasets to illustrate the above scenarios.

``` py
from chronoepilogi import ChronoEpilogi
from ucimlrepo import fetch_ucirepo
```

### One-step-ahead prediction on MTS data

