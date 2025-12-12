# Video Script: Ant Colony Optimization for Multi-Robot Coverage

## Duration: 2.5 minutes (~375 words)

---

### [0:00-0:15] Introduction

**Visual: Show the grid with robots and obstacles**

"Today, we're solving a multi-robot coverage problem using Ant Colony Optimization, or ACO for short.

Imagine you have three robots that need to cover a 6 by 6 grid. Some cells have obstacles that robots can't visit. The goal is to make all three robots work together to cover every free cell efficiently."

---

### [0:15-0:35] How ACO Works - The Big Idea

**Visual: Show ants finding paths, then transition to robots**

"ACO is inspired by how real ants find food. When ants walk, they leave a chemical trail called pheromone. Other ants follow strong trails, making them stronger over time.

In our problem, we use 'virtual ants' to explore different ways of assigning cells to robots. Each ant creates a solution by:

- First, dividing cells among robots using DARP - that's like deciding which robot covers which area
- Then, building paths using UF-STC - that's like finding the best route for each robot to cover its assigned cells"

---

### [0:35-1:05] Desirability Function

**Visual: Show formula and visual representation of distance + workload balance**

"Now, here's the clever part - how do ants decide which cells to assign to which robot? We use something called 'desirability.'

Desirability combines two things: how close a cell is to the robot's starting position, and how balanced the workload is. The formula is:

Desirability equals 1 divided by distance plus gamma times workload imbalance.

So if a cell is close to the robot AND assigning it keeps workloads balanced, it has high desirability. Gamma controls how much we care about balance versus distance."

---

### [1:05-1:30] Probability Calculation

**Visual: Show probability formula with pheromone and desirability**

"When an ant decides which robot should get a cell, it uses probability based on two things: pheromone strength and desirability.

The probability is proportional to pheromone raised to alpha, times desirability raised to beta.

Alpha controls how much we trust past experience - high alpha means we follow strong pheromone trails more.

Beta controls how much we trust the desirability heuristic - high beta means we prefer closer, more balanced assignments."

---

### [1:30-1:55] Pheromone Update

**Visual: Show evaporation and deposit process**

"After all ants finish exploring, we update the pheromone trails. This happens in two steps:

First, evaporation - all pheromone decreases by a factor of rho. This is like the trails fading over time, so old information doesn't dominate forever.

Then, deposit - each ant deposits pheromone on cells it visited. We use the Ant Quantity Model, where each ant deposits an amount equal to the desirability value of that cell.

So good cells - ones that are close and help balance workload - get more pheromone deposited. This makes future ants more likely to choose them."

---

### [1:55-2:15] The Iteration Process

**Visual: Show iteration counter, convergence graph**

"We repeat this whole process 50 times. Each iteration:

- 15 ants explore different solutions
- We calculate desirability for each cell-robot pair
- Ants use probability to assign cells to robots
- We build paths using UF-STC
- We update pheromone trails - evaporate then deposit
- We track the best solution found so far

After 50 iterations, we have our best solution."

---

### [2:15-2:35] Case Study 2 Results

**Visual: Show the results table and visualization**

"Let's look at Case Study 2 results. We ran the algorithm 4 times to check reliability.

Results show:

- 100% coverage - all 30 free cells were covered
- Workload imbalance of 12.00 - meaning robots share work fairly evenly
- Average runtime of 7.3 seconds - very fast!

The best solution found has Robot 0 covering 11 cells, Robot 1 covering 21 cells, and Robot 2 covering 19 cells. All paths are valid - no obstacles hit, no jumps between non-adjacent cells."

---

### [2:35-2:40] Conclusion

**Visual: Show final solution visualization**

"In just 7 seconds, ACO found an excellent solution using pheromone trails, desirability functions, and smart probability calculations. The algorithm successfully balances coverage and workload distribution across all three robots."

---

## Key Visuals Needed:

1. Grid visualization with obstacles (red) and robot paths (different colors)
2. Ants walking and leaving pheromone trails (animated)
3. Desirability formula: η = 1 / (dist + γ|L_r - L̄|)
4. Probability formula: P ∝ τ^α × η^β
5. Pheromone update visualization (evaporation + deposit)
6. Convergence graph showing improvement over iterations
7. Results table with KPIs
8. Final solution showing robot paths on the grid

## Key Formulas:

- **Desirability**: η = 1 / (dist + γ|L_r - L̄|)
- **Probability**: P ∝ τ^α × η^β
- **Pheromone Update**: τ = τ × (1-ρ) + η
- **Parameters**: α=1.0, β=1.0, γ=1.0, ρ=0.5

## Speaking Pace:

- Approximately 150 words per minute
- Total script: ~375 words
- Duration: ~2.5 minutes
