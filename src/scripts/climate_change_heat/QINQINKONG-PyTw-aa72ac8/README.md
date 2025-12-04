# PyTw
This repository contains Cython source code for calculating wet-bulb temperature (Tw) using the Davies-Jones (2008) [1] approach. It is ported from the Fortran code written by Buzan et al (2015) [2] (https://github.com/jrbuzan/HumanIndexMod_2020). The recently identified errors or inappropriate humidity approximations within the Fortran code have been corrected.


### What is in this repository?
- `Cython source files (```wetbulb.pyx``` file) for calculating Tw; Cython source file needs to be compiled first to generate shared object files (```.so``` file) that can be directly imported in Python. Setup tools (```setupwetbulb.py``` file) are recommended for building Cython source file.
- `Calculate_Tw_with_CMIP6_data.ipynb`: A jupyter nobtebook introducing the usage of the code. 
- `environment.yml` a YAML file that can be used to build conda environment containing all needed python packages.

****
### How to use the Jupyte notebooks
Before using the Jupyter notebook, users need to install dependent Python packages listed in `environment.yml`, and compile the Cython source file to generate the shared object file. Users can compile Cython source files with the following command:
- for Intel compiler: 
  - `LDSHARED="icc -shared" CC=icc python setupTw.py develop`; 
- for gcc compiler:
  - replace `-qopenmp` with `-fopenmp` in `setupwetbulb.py` file
  - `python setupTw.py build_ext --inplace`;
  
For introduction to Cython, please refer to https://cython.readthedocs.io/en/latest/

****
### Citation
If you want to use our code, please consider cite `upcoming`

****
### References

[1] Davies-Jones, R. An Efficient and Accurate Method for Computing the Wet-Bulb Temperature along Pseudoadiabats. Monthly Weather Review 136, 2764–2785 (2008).

[2] Buzan, J. R., Oleson, K. & Huber, M. Implementation and comparison of a suite of heat stress metrics within the Community Land Model version 4.5. Geosci. Model Dev. 8, 151–170 (2015).
