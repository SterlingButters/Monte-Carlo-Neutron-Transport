# MonteCarlo-Neutron-Transport
This repo is designed to model a reactor core 
  
This repo was created to run a Monte Carlo Neutron Transport simulation with a minimal number of inputs that are straight forwward and easy to understand.

# Dependencies
- Plotly (latest version)

# Usage
Just change the value of 'N' and run

# Output
- Code will generate a 3D rendering of the reactor core with traces of the neutron trajectories for original (green) and fission spawned neutrons (red)
- Additionally, the code will output the infinite medium multiplication factor based on data records during runtime

# Development
Additional Developers is much desired as my knowledge of Python is limited and thus my efficiency is limited. 

The project Physics wishlist is as follows:
  - Reflector Implementation
  - Control Rod Implementation
  - Source Rod Implementation
  - Full-Core Compatibility (Based on configuration of different assemblies) (Technically full-core compatible however with limited options and speed)
  
The project Computation wishlist is as follows:
  - Concurrency/Multiprocessing/GPU Acceleration
  
![alt text]()
