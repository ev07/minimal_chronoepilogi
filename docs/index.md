# ChronoEpilogi

ChronoEpilogi is a scalable algorithm for Time Series Selection and Markov Boundaries discovery.

**A focus on scalability**

ChronoEpilogi was built to scale. First described in the 2024 NeurIPS paper ["ChronoEpilogi: scalable time-series variable selection with multiple solutions"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f24e8cc1c1c06a689850ee766a7357b2-Abstract-Conference.html),
the library combines algorithmic design and practical optimizations so you can run principled Time Series Selection on high‑dimensional
multivariate time series.

**Mixed-types multivariate time series**

ChronoEpilogi natively supports heterogeneous datasets (numerical, categorical and count series). Declare per‑group types with the `variable_types` mapping and ChronoEpilogi will:

- adapt modeling/equivalence behavior based on the predicted type ("continuous", "binary", "count") and the chosen  learning model implementation;
- use appropriate independence tests to detect informative series (pearson correlation test for continuous data, anova, kruskal or alexander-govern for categorical data);
- harness existing conditional independence tests for mixed-type data to detect information equivalence between Time Series.

**Interpretability with Multiple Markov Boundaries**

ChronoEpilogi provides interpretability by design. By identifying multiple Markov Boundaries, ChronoEpilogi reveals not just a single set of relevant time series, but all minimal sets that explain the target. This approach uncovers alternative explanations and highlights redundancies or synergies among Time Series, giving practitioners a deeper understanding of the underlying temporal relationships. The algorithm’s output is easy to inspect, making it straightforward to interpret which variables are essential, which are interchangeable, and how different combinations can drive predictions or insights.



## Installation

## Quick Start

## Credits

