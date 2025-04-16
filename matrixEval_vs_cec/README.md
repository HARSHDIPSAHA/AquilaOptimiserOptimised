You don’t provide your own data because you're testing your optimizer, not solving a specific real-world problem.

The benchmark suite provides 30 mathematical functions (e.g., Sphere, Rastrigin, Ackley, etc.).

For every run, your Aquila Optimizer generates random position vectors within a range:

python
Copy
Edit
positions = np.random.uniform(lb, ub, (population_size, dim))
Each of these position vectors is fed into the function:

python
Copy
Edit
fitness[i] = fitness_function(positions[i])  # This is func.evaluate(position[i])
That’s how the fitness value is calculated — it's just the value of the test function at that input point.

