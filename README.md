# NFL Network Analysis (CS4973 Final Project)

## Project Overview

**NFL Network Analysis** applies complex network techniques to NFL tracking and play-by-play data from the 2022 season.  We build both offensive influence networks and defensive dynamic play networks to uncover structural features that correlate with play success.  By computing and analyzing centrality, clustering, density, and robustness metrics, we identify the players and formations most responsible for high-impact plays on both sides of the ball.

## Key Features

- **Influence Networks**:  Nodes represent players, edges weighted by average EPA per co-occurrence; centrality measures highlight top passers, unsung bridge players, and positional leaders.
- **Dynamic Play Networks**:  For each play we sample defender positions at one-second intervals and build complete inverse-distance graphs; aggregated metrics reveal strategies behind strong stops.
- **Comprehensive Metrics**:  Compute weighted betweenness, closeness, eigenvector, Katz, PageRank, global and local clustering coefficients, graph density, vertex strength, and robustness simulations.
- **Statistical Analysis**:  Correlate network metrics with Expected Points Added (EPA), perform ANOVA and pairwise t-tests across play cohorts (good, average, poor) to identify significant differences.

## Code Overview

This repository contains the following key scripts and notebooks:

- **offense_analysis.ipynb & defense_analysis.ipynb**  
  Constructs team-level influence networks from play-by-play data.  It merges player statistics with EPA values, builds a weighted co-occurrence graph, computes centrality and clustering metrics, and visualizes the top influencers and bridge players.

- **EPA_Analysis.py**  
  Processes player tracking data to build dynamic play networks.  It joins tracking frames with EPA, samples at one-second intervals, constructs inverse-distance play graphs, computes a full suite of weighted metrics per snapshot, aggregates to play-level profiles, and performs statistical testing across EPA-based cohorts.

- **get_nfl_data.R**  
  Fetches and saves the 2022 NFL play-by-play dataset using the nflfastR package, outputting a clean CSV for downstream analysis.

