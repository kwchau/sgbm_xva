# SGBM-XVA

This project is a Python demonstrator for the stochastic grid bundling method (SGBM) to solve backward stochastic differential equations (BSDE) 
(see https://arxiv.org/abs/1801.05180), using the particular case of XVA calculation with Black-Scholes model. 
It makes use of parallel computing with the CUDA Python platform (https://developer.nvidia.com/how-to-cuda-python) to increase the efficiency of the algorithm. 
While the functions in this project are tailored to the specific problem of XVA, it can be adapted to general Lipschitz BSDEs with proper modification.

## Motivation

This project is part of the WAKEUPCALL project (https://wakeupcall.project.cwi.nl/) in developing new numerical technique for risk management.
Although BSDE is a popular and powerful tool within the academic mathematical finance community, the interest does not seem to be replicated in the
practical side. Thus we would like to show that an efficient computational tool can be developed for BSDE and it can be applied in complicated financial
problems in this work. Parallel Python is the choice of programming language here for its balance of efficiency and ease of development. 

## Getting Started

The code can be simply downloaded and run.

### Prerequisites

In general, the following Python packages are required to run this code.

```
Numpy
Scipy
Numba
```

In addition, the following tools are needed for the CUDA Python part of this program. However, the non-GPU related functions can be run without them.

```
CUDA Toolkit
pyculib
```

## Code Example

To see the basic performance of the code, user can run the script numerical_test.

```
python numerical_test 
```

The script will ask under what stock dimension should the test be run, the setting of testing parameters and whatever the preset test for GPU solver and CPU solver should be run.
No GPU and related tools are needed if only the CPU part is used. More details on the code can be found in the technical document and the research report.
The test result for each test run will be shown on the display with the average saved in a test file.

Furthermore, user can run test on or given 5 dimansional test example with 

```
from numba_sgbm_xva import numba_sgbm_xva
from numba_blackscholes import numba_monte_carlo

numba_sgbm_xva(numba_monte_carlo, num_of_path, num_of_time_step, num_of_bundle)
```

or

```
from cuda_python_sgbm_xva import cuda_sgbm_xva
from cuda_blackscholes import cuda_jit_montecarlo

cuda_sgbm_xva(cuda_jit_montecarlo, num_of_path, num_of_time_step, num_of_bundle)
```

for pure Numba jit or CUDA Python test respectively.

One can input the following code to change the example dimension.

```
import example_xva_v2
example_xva_v2.Example = example_xva_v2.ArithmeticBasketPut(val)
import example_xva_v2
```

## File Structure
The cuda_python_sgbm_xva and numba_sgbm_xva files contain the main function and the backward portion of the program, 
for CUDA Python version of SGBM and pure Numba jit version of SGBM respectively.
The cuda_blackscholes and numba_blackscholes files contains the forward portion of the code.
The example_xva_v2 files contains the example medel setting and the numerical scheme for BSDE.

## Authors

* **Ki Wai Chau** - *Initial work* 

## License

*  The MIT License

## Acknowledgments

* This project is developed in VORtech with the help of their programmers.
