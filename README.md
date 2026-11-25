# Corporate Credit Rating Transitions with Economic Regime and Momentum-Dependent Markov Models

**STAT 471 — Probability I | University of Alberta | Fall 2026**

**Authors:** Dotun Famobio, John Thompson, and Maksym Yefimenko

## Overview

This project investigates corporate credit rating transitions and default risk using discrete-time Markov chains. We extend the traditional credit rating transition model by incorporating economic regimes and credit rating momentum.

The primary objective is to determine whether incorporating economic conditions and historical rating movements improves the predictive performance of conventional Markov models.

Four models are constructed and compared through analytical probability calculations, Monte Carlo simulations, and historical model evaluation.

## Objectives

- Model corporate credit rating transitions using discrete-time Markov chains.
- Investigate the impact of economic regimes on credit risk.
- Investigate the impact of previous rating movements on future transitions.
- Calculate cumulative probabilities of default over multiple time horizons.
- Validate analytical results using Monte Carlo simulations.
- Compare model predictive performance against historical credit rating data.

## Models

| Model | State Space | Description |
|---|---|---|
| Baseline | $R_t$ | Depends only on the current credit rating. |
| Regime | $(R_t,E_t)$ | Incorporates economic expansion and recession. |
| Momentum | $(R_t,M_t)$ | Incorporates previous credit rating movements. |
| Combined | $(R_t,E_t,M_t)$ | Incorporates both economic regime and momentum. |

### Credit Rating State Space

$$
S = \{AAA, AA, A, BBB, BB, B, CCC/C, D\}
$$

The default state, $D$, is absorbing, meaning that once a company enters default, it cannot transition to another rating.

### Baseline Model

The baseline model assumes that future credit rating transitions depend exclusively on the current credit rating.

$$
P_{ij} = \mathbb{P}(R_{t+1}=j \mid R_t=i)
$$

The baseline transition matrix is constructed using historical corporate credit rating transition probabilities published by S&P Global Ratings.

### Economic Regime Model

The regime-dependent model incorporates the state of the economy into credit rating transitions.

$$
E_t \in \{\text{Expansion}, \text{Recession}\}
$$

Economic regime multipliers are applied to the baseline transition matrix to capture differences in credit rating migration under different economic conditions.

### Momentum Model

The momentum-dependent model accounts for the direction of the previous credit rating transition.

$$
M_t \in \{\text{Upgrade}, \text{Unchanged}, \text{Downgrade}\}
$$

Momentum multipliers are applied to the baseline transition matrix to capture persistence in credit rating movements.

### Combined Model

The combined model incorporates both economic regime and credit rating momentum.

$$
X_t = (R_t,E_t,M_t)
$$

By augmenting the state space, historical rating information and economic conditions can be incorporated while preserving the Markov property of the augmented process.

## Data Sources

| Source | Purpose |
|---|---|
| S&P Global Ratings | Baseline transition probabilities and historical rating information. |
| Federal Reserve Economic Data (FRED) | Historical recession and expansion indicators. |
| Published credit migration research | Momentum-dependent transition probabilities. |
| Fitch Ratings | Historical credit rating transitions for model evaluation. |

Rating categories are standardized across datasets to ensure consistency between the models.

## Methodology

1. Construct the baseline credit rating transition matrix.
2. Obtain historical economic regime information.
3. Construct economic regime multipliers.
4. Construct credit rating momentum multipliers.
5. Apply the multipliers to the baseline transition matrix.
6. Normalize transition matrices to ensure that each row sums to one.
7. Calculate analytical probabilities of default.
8. Conduct Monte Carlo simulations.
9. Evaluate and rank the models against historical rating transitions.

### Analytical Default Probability

For the baseline transition matrix $P$, the cumulative probability of default within $n$ years, given an initial credit rating $i$, is

```math
\mathbb{P}(\tau_D \leq n \mid R_0=i) = (P^n)_{iD}
```

where $\tau_D$ represents the first time the process enters the default state.

Because default is absorbing, the probability of occupying the default state after $n$ years is equivalent to the probability of having defaulted within those $n$ years.

For the extended models, the corresponding conditional or augmented transition dynamics are used.

### Monte Carlo Simulation

Monte Carlo simulations are used to generate possible future credit rating trajectories under each model.

- Simulation horizon: 20 years
- Number of simulated paths: 10,000
- Default is treated as an absorbing state.

The empirical cumulative probability of default is calculated as

$$\widehat{PD}(n) = \frac{1}{N}\sum_{k=1}^{N}\mathbf{1}_{\{\tau_D^{(k)} \leq n\}}$$

where $N$ is the number of simulated paths.

Analytical and simulated probabilities are compared to assess the numerical consistency of each model.

## Historical Model Evaluation

The four models are evaluated against observed historical corporate credit rating transitions from 2016 to 2023.

Rather than training new models, the existing transition probabilities are evaluated against historical observations.

Two evaluation metrics are considered:

### Root Mean Squared Error (RMSE)

Measures the difference between predicted and observed default rates.

$$
RMSE =
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
(\widehat{PD}_t-PD_t)^2
}
$$

### Negative Log-Likelihood (NLL)

Measures how well a model assigns probabilities to observed credit rating transitions.

$$
NLL =
-\frac{1}{N}
\sum_{k=1}^{N}
\log(p_k)
$$

where $p_k$ is the predicted probability of the observed transition.

Lower values indicate better predictive performance for both metrics.

## Results

### 20-Year Cumulative Default Probability

Initial credit rating: BB

| Model | Analytical | Monte Carlo |
|---|---:|---:|
| Baseline | 32.85% | 32.43% |
| Regime | 31.58% | 30.97% |
| Momentum | 34.91% | 34.88% |
| Combined | 33.57% | 33.94% |

The analytical and Monte Carlo estimates agree within one percentage point across all four models, supporting the numerical consistency of the implementation.

### Historical Default Prediction Ranking

| Rank | Model | Default RMSE |
|---|---|---:|
| 1 | Momentum | 0.3630% |
| 2 | Baseline | 0.3880% |
| 3 | Combined | 0.5787% |
| 4 | Regime | 0.5831% |

The momentum model achieves approximately a 6.4% reduction in default prediction RMSE relative to the baseline model.

However, the baseline model achieves the lowest mean negative log-likelihood, indicating superior overall credit rating transition prediction in the historical evaluation.

## Key Findings

- Economic recessions and recent credit rating downgrades are associated with higher modeled conditional default risk.
- Analytical and Monte Carlo default probabilities are closely aligned across all four models.
- Incorporating credit rating momentum improves historical default prediction relative to the baseline model.
- The baseline model performs best when predicting overall credit rating transitions.
- Incorporating additional information and increasing model complexity does not necessarily improve predictive performance.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_processing.py
│   ├── transition_matrix.py
│   ├── fred_loader.py
│   ├── economic_regimes.py
│   ├── regime_multipliers.py
│   ├── momentum_multipliers.py
│   ├── regime_model.py
│   ├── momentum_model.py
│   ├── combined_model.py
│   ├── simulation.py
│   ├── metrics.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the main analysis notebook in the `notebooks/` directory and execute the cells sequentially.

Ensure that the required datasets are available in the appropriate data directories before running the notebook.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Jupyter Notebook
- FRED economic data

## Limitations

- Limited historical recession observations may affect the reliability of economic regime multipliers.
- U.S. economic recession indicators may not fully represent global economic conditions.
- Momentum multipliers are derived from published historical data and may require approximations across rating categories.
- The combined model assumes that regime and momentum effects can be incorporated multiplicatively.
- Historical evaluation is restricted to a particular observation period and is not necessarily a strictly out-of-sample test.
- Additional model complexity introduces assumptions that may not accurately reflect real-world credit rating dynamics.

## Future Work

- Estimate transition probabilities directly from issuer-level historical data.
- Incorporate additional macroeconomic variables, such as interest rates, inflation, and unemployment.
- Investigate continuous-time Markov chain formulations.
- Explore alternative methods for estimating regime and momentum effects.
- Conduct more extensive out-of-sample historical validation.
- Investigate statistical uncertainty in estimated default probabilities.

## Conclusion

Economic regimes and credit rating momentum provide additional insight into corporate credit risk and rating migration.

Although state space augmentation allows Markov models to incorporate more information, the historical evaluation demonstrates that increased model complexity does not guarantee improved predictive performance.

The momentum-dependent model performs best for historical default prediction, while the baseline model performs best for overall credit rating transition prediction.

These findings highlight the importance of balancing model complexity, interpretability, and empirical predictive performance in quantitative credit risk modeling.

## Authors

- Dotun Famobio
- John Thompson
- Maksym Yefimenko

Department of Mathematical and Statistical Sciences  
University of Alberta  
Fall 2026

## Disclaimer

This project was developed for academic and research purposes and is not intended to provide investment advice or serve as a production credit risk assessment system.
