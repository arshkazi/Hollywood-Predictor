# Graph-Powered Box Office Forecasting: Hybrid Social Network Analysis and Machine Learning Pipeline

This repository contains a comprehensive, production-grade machine learning pipeline that forecasts global theatrical box office revenue before a film begins physical production.

By integrating **Social Network Analysis (SNA)** with **Ensemble Regression Techniques**, this system moves beyond traditional, static categorical encoding. It models Hollywood as a living, evolving co-occurrence network where the structural position of creative talent dictates historical leverage and financial draw.

The pipeline achieves an **R-squared ($R^2$) score of 0.4342**, explaining nearly half of all global box office variance using strictly pre-production inputs.

---

## Technical Methodology: Architectural Highlights

Predicting theatrical performance is notoriously volatile. Traditional approaches rely on simple categorical dummy variables (e.g., One-Hot Encoding the names of actors or directors). This project explicitly rejects that approach because:

1. **High Cardinality:** Thousands of unique actors and directors create a sparse, high-dimensional feature space that causes overfitting (the curse of dimensionality).
2. **Loss of Collaborative Context:** Dummy variables treat "Tom Cruise" and "John Woo" as completely independent entities, failing to capture their history of shared high-grossing projects or their combined structural "pull" in the industry.

To resolve these limitations, this pipeline implements a two-stage hybrid model:

```
[ Phase 1: Bipartite Graph Construction ]
  Create nodes for movies, actors, and directors. Connect talent to their respective films.
                       │
                       ▼
[ Phase 2: Structural PageRank Centrality ]
  Run Google's PageRank over the talent network to project graph topology into a single, continuous numerical metric.
                       │
                       ▼
[ Phase 3: Feature Scaling & Data Partitioning ]
  Apply StandardScaler to synchronize high-magnitude budgets with fractional PageRank centralities.
                       │
                       ▼
[ Phase 4: Ensemble Learning ]
  Train a Random Forest Regressor to capture non-linear feature interactions (e.g., Star Power × Budget scaling).

```

---

## Data Engineering and Cleaning Pipeline

The system ingests and unifies two primary, high-discrepancy data sources:

* **The TMDB Dataset:** Contains granular pre-production metadata (budget, runtime, release date, cast list, and director) for 9,999 films.
* **The Enhanced Box Office Dataset (2000–2024):** Contains verified, inflation-adjusted historical worldwide gross records for 5,000 films.

```python
# Create standardized, alphanumeric matching keys to resolve case sensitivity and whitespace discrepancies
tmdb_df['title_clean'] = tmdb_df['title'].astype(str).str.strip().str.lower()
box_office_df['title_clean'] = box_office_df['Release Group'].astype(str).str.strip().str.lower()

# Inner merge to ensure every record has verified ground-truth targets paired with complete TMDB metadata
merged_df = pd.merge(box_office_df, tmdb_df, on='title_clean', how='inner', suffixes=('_box', '_tmdb'))

```

### Explicit Inclusions and Data Filtering Rules:

* **The $100k Threshold Rule:** Records with production budgets or worldwide grosses under $100,000 are systematically dropped. This filters out student projects, short films, straight-to-streaming titles, and micro-budget indie releases, aligning the training distribution with commercial theatrical releases.
* **Strict Runtime Bounds:** Only films with complete, non-null runtimes equal to or greater than 60 minutes are retained.
* **Temporal Cleaning:** Release date strings are parsed into standardized datetime objects to handle formatting anomalies. Rows with missing temporal metadata are discarded.

---

## Social Network Analysis: Graph Construction and Feature Engineering

Instead of treating the cast and crew as static text, we project them into an undirected, bipartite-like collaboration network using the `networkx` library.

### Why PageRank Centrality?

Simple centrality metrics like Degree Centrality (counting the number of movies an actor has worked on) are flawed. An actor with 10 minor roles in obscure films would score higher than an actor with 5 lead roles in massive, career-defining blockbusters.

By applying Google's **PageRank algorithm** over the global network, we calculate a recursive influence metric:


$$PR(u) = \frac{1-d}{N} + d \sum_{v \in B_u} \frac{PR(v)}{L(v)}$$


Where $PR(u)$ is the star-power score of talent $u$, $d$ is a damping factor (set to default 0.85), $B_u$ is the set of all movies linking to $u$, and $L(v)$ is the connection density of movie $v$. This mathematically guarantees that an actor's "Star Power" increases if they work with highly connected directors, who in turn are connected to other highly connected actors.

```python
import networkx as nx
import ast

# Initialize an undirected network graph
G = nx.Graph()

# Build nodes and map collaborative edges
for idx, row in cleaned_df.iterrows():
    movie_node = f"MOVIE_{row['title_clean']}"
    G.add_node(movie_node, type='movie')
    
    # Parse stringified lists safely using Abstract Syntax Trees
    cast_list = ast.literal_eval(row['cast']) if isinstance(row['cast'], str) else row['cast']
    director_list = ast.literal_eval(row['director']) if isinstance(row['director'], str) else row['director']
    
    for actor in cast_list:
        actor_node = f"ACTOR_{actor.strip().lower()}"
        G.add_node(actor_node, type='actor')
        G.add_edge(movie_node, actor_node) # Connect actor node to movie node
        
    for director in director_list:
        director_node = f"DIRECTOR_{director.strip().lower()}"
        G.add_node(director_node, type='director')
        G.add_edge(movie_node, director_node) # Connect director node to movie node

# Compute global PageRank vectors across the entire industry topology
pagerank_scores = nx.pagerank(G)

```

Once computed, these continuous PageRank vectors are mapped back to each movie record to create two high-signal features:

* `cast_star_power`: The sum of the PageRank scores of the billed actors.
* `director_star_power`: The PageRank score of the lead director.

---

## What We Are Using vs. What We Are Not Using

Understanding the architectural constraints of this pipeline requires looking closely at what features and techniques were deliberately left out:

### What We Are Using:

* **Continuous Financial Inputs (`budget`):** Raw production funding, standardized to handle high-magnitude values.
* **Temporal Seasonality Flags (`is_summer`, `is_holiday`):** Binary indicators representing peak theatrical windows (June–August and November–December) to model historical theater attendance patterns.
* **Physical Properties (`runtime`):** Continuous length of the film, which correlates heavily with theater rotation frequency and narrative scale.
* **Graph Topological Features (`cast_star_power`, `director_star_power`):** Quantified mathematical representation of creative capital and industry influence.
* **Robust Feature Scaling (`StandardScaler`):** Standardizes all features to a mean of 0 and variance of 1. This prevents high-value features (like a $200 million budget) from numerically dominating fractional PageRank scores (like 0.0012) during optimization.

### What We Are NOT Using (By Architectural Design):

* **One-Hot Encoding on Names:** Banned to prevent catastrophic column explosions and massive, unmanageable sparse matrices.
* **Textual Description/Plot Embedding:** Natural Language Processing (NLP) on movie plot synopses is excluded. While creative, synopsis text rarely correlates with theatrical box office success before a film is produced.
* **Studio Name Categorization:** Excluded because distributor and production company names change rapidly due to corporate acquisitions and shell production LLCs, introducing high noise into the dataset.
* **Review Scores / Post-Release Metrics:** The model explicitly avoids using `vote_average` or `vote_count`. These are post-release metrics; using them would create severe data leakage, rendering the pre-production forecasting model useless for early greenlight decisions.

---

## Machine Learning & Model Performance

The dataset is partitioned using an **80/20 train-test split**, ensuring that performance metrics are evaluated strictly on unseen data.

We benchmark a baseline linear estimator against a non-linear ensemble tree-based regressor:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Fit the Linear baseline
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Fit the non-linear Random Forest Ensemble
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_scaled, y_train)

```

### Performance Metrics Comparison

| Model | Test MAE (Mean Absolute Error) | Test R-squared ($R^2$) |
| --- | --- | --- |
| **Linear Regression** | $112,578,914.15 | 0.4287 |
| **Random Forest Regressor** | **$108,940,698.65** | **0.4342** |

The **Random Forest Regressor** out-performs the linear baseline because it successfully resolves multi-variable thresholds. For example, a high-budget film released in a non-holiday month behaves differently than a low-budget film released in the same period; a linear model struggles to map these interactive, conditional relationships, whereas decision trees segment them naturally.

---

## Feature Importance and Interpretability

By computing the mean decrease in impurity across the forest's ensemble of decision trees, we extract the exact mathematical weight of each feature:

```
Rank  Feature                 Relative Weight (%)
─────────────────────────────────────────────────
1     BUDGET                  ██████████████████ 52.77%
2     CAST_STAR_POWER         ██████ 16.64%
3     DIRECTOR_STAR_POWER     █████ 15.07%
4     RUNTIME                 ████ 12.40%
5     IS_SUMMER               █ 1.91%
6     IS_HOLIDAY              █ 1.21%

```

### Critical Insights:

* **The Power of Connection (31.71% Combined):** Our custom-engineered PageRank metrics (`CAST_STAR_POWER` and `DIRECTOR_STAR_POWER`) hold almost a third of the model's total decision-making power. This mathematically validates our social network analysis approach.
* **Budget Dominance (52.77%):** Budget remains the primary driver of box office performance, reflecting the sheer scale of marketing, distribution, and screen allocation that large investments unlock.

---

## Code Execution: Step-by-Step

### 1. Requirements and Setup

Ensure you have the required dependencies installed in your Python environment:

```bash
pip install pandas numpy networkx scikit-learn

```

### 2. Execution Pipeline

Create a script named `pipeline.py` and run the entire sequence:

```bash
python pipeline.py

```

The script will ingest the raw CSV files, build the network graph, calculate PageRank values, scale the training features, output the performance metrics, and generate test predictions.
