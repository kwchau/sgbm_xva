"""
Created on Tue May 29 12:07:51 2018

@author: kiwai
"""
import logging

import example_xva_v2 

def integer_check(welcome, limit):
    while True:
        number = input(welcome)
        try:
            val = int(number)
            if val < limit:  # if not a positive int print message and ask for input again
                print("Sorry, input must be at least equal to " + str(limit)+ ", try again")
                continue
            break
        except ValueError:
            print("That's not an int!")     

    return val



# This ask the user if the GPU test should be run.
while True:
    val = integer_check("Pick the dimension for the stock model:", 1)
    example_xva_v2.Example = example_xva_v2.ArithmeticBasketPut(val)
    import example_xva_v2
        
    from numba_sgbm_xva import numba_sgbm_xva
    from numba_blackscholes import numba_monte_carlo 

    print("This script will run a SGBM pricing test under the setting of (paths, time step, bundles)= (2^(14 + i), 4*(i+1), 2^(3+i))")   
    start = integer_check("Pick starting point for i's range:", 0)
    end = integer_check("Pick end point for i's range:", start + 1)
    test = input("Perform the GPU test(T/F):")
    if test == "T":
        GPU = True
        break
    elif test == "F":
        GPU = False
        break
    else:
        print("Invalid input\n")

# This ask the user if the CPU test should be run.
while True:
    test = input("Perform the Numba test(T/F):")
    if test == "T":
        CPU = True
        break
    elif test == "F":
        CPU = False
        break
    else:
        print("Invalid input\n")

#  The GPU related functions are only imported when the GPU test is run, as CUDA library is required 
#  when importing these functions
if GPU:
    from cuda_python_sgbm_xva import cuda_sgbm_xva
    from cuda_blackscholes import cuda_jit_montecarlo

    NUMBA_DEBUGINFO=0
    
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.INFO)

with open('data.txt','a') as data: 
    data.write('Numerical test for ' + example_xva_v2.Example.name + ' with stocks dimension as ' + str(example_xva_v2.Example.dimension) + '\n')

#  Perform the hard-coded test
for i in range(start, end):
    num_of_path = 2 ** (14 + i)
    num_of_time_step = 4 + 4 * i
    num_of_bundle = 2 ** (3 + i)
    if GPU:
        table = cuda_sgbm_xva(cuda_jit_montecarlo, num_of_path, num_of_time_step, num_of_bundle)
        with open('data.txt','a') as data: 
            data.write('Test for CUDA with paths: '+str(num_of_path) + ' steps: ' + str(num_of_time_step) + ' bundle: ' + str(num_of_bundle)+ ' ')
            data.write(str(table)+'\n')
    if CPU:
        table = numba_sgbm_xva(numba_monte_carlo, num_of_path, num_of_time_step, num_of_bundle)
        with open('data.txt','a') as data: 
            data.write('Test for Numba with paths: '+str(num_of_path) + ' steps: ' + str(num_of_time_step) + ' bundle: ' + str(num_of_bundle)+ ' ')
            data.write(str(table)+'\n')
    print('Finish Run ' + str(i))
