# GT-ics – The Evolution of Cooperation

This project models and simulates the **evolution of cooperation** using the Iterated Prisoner's Dilemma and Genetic Algorithms. It explores how cooperation can emerge among selfish agents in repeated interactions.

## Project Summary

We simulate a population of agents using different strategies to compete in iterated Prisoner’s Dilemma tournaments. Strategies evolve over generations using a genetic algorithm that applies selection, crossover, and mutation to generate new, potentially more successful strategies.

## Goals

- Understand how cooperative behavior can evolve in competitive environments.
- Implement a GUI to explore parameter tuning and visualize tournament dynamics.
- Analyze performance of different strategies through simulations and statistical experiments.
- Evolve new strategies via genetic algorithms and compare them against classic ones like Tit-for-Tat.

## Key Features

- 10 custom-built deterministic strategies (rule tables).
- Genetic algorithm to evolve strategies.
- Visualizations of population dynamics, convergence, and strategy fitness.
- Fully interactive GUI with tunable parameters.
- Experiments on:
  - Effect of game length
  - Alternative payoff matrices
  - Stability and robustness of evolved strategies

## Model Overview

- **Base Game**: Iterated Prisoner’s Dilemma
- **Strategies**: Encoded as rule tables that map interaction history to actions (C or D)
- **Payoffs**: Configurable matrix, e.g.:

|            | B Cooperates | B Defects |
|------------|--------------|-----------|
| A Cooperates | 3, 3        | 0, 5     |
| A Defects    | 5, 0        | 1, 1     |

- **Genetic Algorithm Steps**:
  1. Initialize population of random strategies
  2. Evaluate fitness through tournaments
  3. Select top performers
  4. Apply crossover and mutation
  5. Repeat until convergence

