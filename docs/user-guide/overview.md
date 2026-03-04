# Overview of ChronoEpilogi

This page summarizes some key elements of ChronoEpilogi.
For an in-depth description of the algorithm, we refer to our paper [1].

## Motivation: modeling large Multivariate Time Series

ChronoEpilogi is a Time Series Selection (TSS) algorithm : given a Multivariate Time Series (MTS), it selects appropriate dimensions to forecast a quantity of interest. 

For instance, we have a MTS $X$ of size $T\times D$, with $T$ timestamps and $D$ Time Series recorded. We are interested in forecasting one of the dimensions that we will call the target TS $X^1$, considering a lookback windows of $L$ lags at horizon 1. Formally, we want to learn a model $f$ to approximate $\mathbb{E}[X^1_t | X^{1:D}_{t-L...t-1}]$. This problem is called *one step ahead point forecasting*. ChronoEpilogi extends to other similar tasks, see [data and tasks](..\\..\\user-guide\\data_and_tasks).

Generally, using $X^{1:D}_{t-L...t-1}$ is unadvisable when $D$ is large: models trained on high dimensional data are less efficient and more likely to overfit. Only a portion of the MTS $D$ dimensions might be useful for the forecasting task, and finding minimal, optimally predictive sets of TS is ChronoEpilogi's function. We distinguish Single TSS and Multiple TSS tasks. 

<div style="border: 2px solid #0078D4; border-radius: 8px; background: #f3f7fa; padding: 1em; margin-bottom: 1.5em;">
<b>Single TSS</b>:  Find a maximally predictive, minimal in size set of TS to forecast one of the TS in the MTS.<br>
<b>Multiple TSS</b>: Find all such sets.
</div>

Single TSS is useful to set of TS optimally suited for forecasting. Multiple TSS produces different sets of TS, which offer analysts a freedom of choice in which set is most suitable to use, given that they are predictively equivalent but may have different costs for the user. We also provide the different TS sets in a factorized form, called equivalence classes of TS. Each equivalence class contains TS that have equivalent information on the target TS, meaning that they can interchangeably be included in a TSS solution. Studying equivalences give knowledge on the structure of the MTS, hence providing interpretability.


## ChronoEpilogi phases: single and multiple Time Series Selection

We provide utilities to solve both problems, through the keyword argument `phases` of the [ChronoEpilogi class](..\\..\\api\\chronoepilogi). The phases are consecutive: we first obtain a single solution (hence solve the Single TSS problem) then build equivalent solutions (TSS multiple).

The single TSS variant of ChronoEpilogi can be accessed with `ChronoEpilogi(phases="FB")`. It includes a *Forward* phase and a *Backward* phase. The forward phase iteratively builds a maximally predictive set of TS starting from an empty set. The backward phase eliminates potentially redundant TS from this set afterwards, ensuring that the returned set `ChronoEpilogi.get_first_markov_boundary()` is both minimal and maximally predictive.

The multiple TSS variant can be discovered with `ChronoEpilogi(phases="FBEV")`. After the first solution is found, the *Equivalence* phase looks for TS that can be swapped with TS of the first solution, while keeping predictive performance maximal. As a result, ChronoEpilogi builds an equivalence class of TS for each TS in the first solution. Other solutions can be obtained by picking one TS per equivalence class. After a *Verification* phase where false positive equivalences are removed, `ChronoEpilogi.get_equivalence_classes()` returns the equivalence classes.

## A modular Time Series Selection algorithm

ChronoEpilogi relies on several modules that manipulate the data. Three modules are necessary for equivalence detection, the first two (Learning model and Association) are required for the *Forward* and *Backward* phases (single TSS problem).

While advanced use of ChronoEpilogi may require digging into each module, ChronoEpilogi has default configuration parameters which automatically choose appropriate ones depending on the type of the forecast quantity, the type of predictors and the data format.

### Learning models

ChronoEpilogi relies on predictive models to assess whether a TS is beneficial to the forecasting task. The learning model has two functions:

1) To produce residuals (the errors of the model) when using a certain set of TS as input. Residuals contain information that "remains to be modeled", hence requires additional predictors. Both TSS-single and TSS-multiple variants depend on the residuals.
2) To enable statistical testing of model equivalence. This provides a method to end the *Forward* phase, remove conditionally independent (aka redundant) TS in the *Backward* phases, and possibly find equivalence classes during the *Equivalence* and *Verification* phases.

By default, ChronoEpilogi uses a linear model suited to the data at hand: a simple linear model for continuous targets, a Poisson GLM for count TS and a Logit GLM for binary TS.

In case that the provided models prove unsuitable for the data at hand, the [LearningModel class](..\\..\\api\\forecasting_models) is a wrapper that specifies what ChronoEpilogi main class expects from a learning model. Subclassing `LearningModel` requires filling the `__init__`, `fit` (model training/fitting), `fittedvalues` (returning predictions) and `stopping_metric` (testing for model equivalence) methods.

### Associations

During the *Forward* phase of the algorithm, we build models by iteratively adding TS to our selected set. How to choose the next TS to include is the responsibility of the Association. 
Given a vector of model residuals (the information to model) and the remaining candidate TS, the Association produces a score for each TS, the higher the more informative the TS is to the residuals.

By default, ChronoEpilogi uses pearson correlation for continuous (`numerical`) TS, and one-way ANOVA for binary and count (`categorical`) TS as Association, depending on the provided TS types.

We note that the association is also used in the equivalence phase if ChronoEpilogi's `equivalence_early_stopping` is set to `True`, which it is by default.


### Partial correlations

Searching for equivalent TS during the *Equivalence* phase is a lengthy process. We introduced heuristics to lessen computation time. The best empirical results were obtained using a partial correlation method, which requires its own module. 

Similarily to the Association, the default parameter setting of ChronoEpilogi will choose the partial correlation appropriate for the data. For continuous values, pearson partial correlation is used. For mixed data, we use a mixed-type linear conditional independence test implemented in the [tigramite library](https://github.com/jakobrunge/tigramite/tree/master).

## References

[1]. ["ChronoEpilogi: scalable time-series variable selection with multiple solutions"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f24e8cc1c1c06a689850ee766a7357b2-Abstract-Conference.html).