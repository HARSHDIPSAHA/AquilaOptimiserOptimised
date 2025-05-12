
https://www.overleaf.com/project/67ef9b047b511c4f9aa94c7f

https://docs.google.com/document/d/1ciTjYeQDRxkatJn0xC884oHGVLI6fWA06ScVH8-RpS0/edit?tab=t.0


# Aquila Optimiser Optimised

This repository contains implementations and analyses of optimization algorithms, including the Aquila Optimizer (AO) and its variants. It provides a comprehensive comparison of these algorithms using benchmark functions and statistical evaluations.

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

## Features

- Implementation of multiple optimization algorithms.
- Statistical evaluation and comparison of algorithms using benchmark functions.
- Clear documentation and structured organization of code.

## Folder Structure

The repository is structured as follows:

```
AquilaOptimiserOptimised/
├── 7algos/
│   ├── AO.py                  # Implementation of the Aquila Optimizer
│   ├── EO.py                  # Implementation of the Equilibrium Optimizer
│   ├── GOA.py                 # Implementation of the Grasshopper Optimization Algorithm
│   ├── PSO.py                 # Implementation of the Particle Swarm Optimization
│   ├── tempCodeRunnerFile.py  # Temporary code runner file
├── MVAO/
│   ├── 60kval(modified).ipynb # Jupyter notebook for a modified version of the Aquila Optimizer
├── Variants_Comparison/
│   ├── Modified.py            # Comparison of modified algorithm variants
│   ├── wilcoxon_results.csv   # Statistical results for Wilcoxon tests
├── matrixEval_vs_cec/
│   ├── README.md              # Notes on benchmark evaluations
│   ├── wilcoxon_results_all_pairs.csv # Statistical results for all pairs
```

### Detailed File Descriptions

#### `7algos/`
Contains implementations of various algorithms:
- **AO.py**: Aquila Optimizer implementation.
- **EO.py**: Equilibrium Optimizer implementation.
- **GOA.py**: Grasshopper Optimization Algorithm implementation.
- **PSO.py**: Particle Swarm Optimization implementation.
- **tempCodeRunnerFile.py**: Temporary Python file for code testing.

#### `MVAO/`
- **60kval(modified).ipynb**: Jupyter notebook demonstrating a modified version of the Aquila Optimizer.

#### `Variants_Comparison/`
- **Modified.py**: Python file comparing modified algorithm variants.
- **wilcoxon_results.csv**: CSV file containing Wilcoxon statistical test results for comparing algorithms.

#### `matrixEval_vs_cec/`
- **README.md**: Additional notes and details on benchmark evaluations.
- **wilcoxon_results_all_pairs.csv**: CSV file with Wilcoxon results for all function pairs.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HARSHDIPSAHA/AquilaOptimiserOptimised.git
   ```

2. Navigate to the folder containing the desired algorithm:
   ```bash
   cd 7algos
   ```

3. Run the Python file for the selected algorithm. For example, to run the Aquila Optimizer:
   ```bash
   python AO.py
   ```

4. To view benchmark evaluations or statistical analyses, navigate to the respective folders.

## Dependencies

Ensure you have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `scipy`

Install dependencies via pip:
```bash
pip install numpy matplotlib scipy
```

## References

- **Aquila Optimizer**: Inspired by the hunting strategies of Aquilas.
- **Benchmark Functions**: Mathematical functions used for evaluating optimization algorithms.
- **Wilcoxon Test**: Statistical test for comparing paired data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to explore the repository to learn more about the implementations and analyses.
