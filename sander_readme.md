## Readme Sander

## General
The code is stored in 2 files, one containing the MPS and TimeOperator objects, and the simulation routines, and one containing some initialization methods for the MPS objects
These files should be stored in the same directory.
The following libraries are required: numpy, scipy, matplotlib, pickle, os, datetime and time. These should all come standard with python 3.7.
After simulation, you can elect to save an MPS object for later reference, which is done using pickle. Currently, you should have a folder named "data" in your working directory in which they can be stored.
MPS objects are stored as "MonthDay_HourMinute_Name_Chainlength_Truncparameter.pkl". Sorting the data folder by alphabetical thus also automatically sorts it by date.

## On how the code works
The code_sander.py module mainly contains 2 large objects. I have elected at the start to work object oriented, but the objects have become a bit large.
The first object is "MPS", which stores all information of an MPS, routines to apply operators, calculate norms and a method for time evolution.
The second object is "TimeOperator", which stores a Hamiltonian and possible dissipative terms, as well as their matrix exponentials or the Crank-Nicolson approximation thereof.
Latsly, there are some routines related to the "NORM_state". The NORM_state is a simple form of the vectorized identity matrix.
In the main function, necessary MPS and TimeOperator objects are created and initialized or loaded from memory, time evolution is performed, and if desired the time evolved MPS is stored again.

### The MPS object
The MPS object contains all defining information, matrices and routines of the MPS. An important boolean in the initialization is the "is_density" bool, which is used to define if an MPS is a 'regular' MPS or a vectorized density matrix.
The objects naturally have a name. A 'regular' MPS is an MPS of a state vector, and is referred to as MPS1 or MPS2. An MPS of a vectorized density matrix is referred to as DENS1 or DENS1. From hereon I will use these terms to refer to the different types of objects.
The is_density boolean is used throughout the code in case routines differ for MPS or DENS type objects. Note that the NORM_state is a DENS type object.
The other attribute of note is the "locsize" attribute. As you may or may not know, the non-zero part of the Lambda and Gamma matrices grows the closer it is to the center of our MPS, up to a maximal size chi which we choose.
The locsize array stores the maximum relevant parts of the Lambda_mat and Gamma_mat arrays, which are all square chi by chi matrices for convenience.
So, Gamma_mat i can be nonzero in the slice :locsize i, :locsize (i+1.
The rest of the MPS object is simple tensor contraction routines. We will treat TEBD and the time evolution later.

### The TimeOperator object
This object also contains an is_density boolean, as the Hamiltonians used for time evolution differ for MPS and DENS type objects.
The Hamiltonians are hardcoded, in this case an XXZ model with a possible transverse magnetic field in the z direction.
The singlesite transverse terms are evenly split between the two parts of the trotter decomposition, with half in the even part and half in the odd part.
After initializing the Hamiltonian, the time evolution operator is constructed. This is done either by directly calculating the exact exponential or by using a Crank-Nicolson approximation.
Next, if specified through the boolean "diss_bool", the dissipative terms are constructed, which is slightly more involved.
The function "calculate_diss_site" simply takes a desired operator and calculates the term that we will later place in the exponential (the second line in equation 4 of Jaschke, Montangero and Carr, 2018).
Of course, we can have more than one Lindblad operator acting on a site. If we give "calculate_diss_site" an array or list of operators, it will sum all these contributions.
The function "calculate_diss_timeop" again creates the exponential or Crank-Nicolson operator of the dissipative terms.
The function "create_diss_array" creates a hardcoded 'pseudo object' with two linked attributes. As the dissipative terms act on specific sites, this function creates a matrix with three rows, "diss_arr".
"diss_arr" is an array with three attributes: index, operator and timeop.
"Index" is the site these terms act on, "operator" is the Lindblad operator or list/array of Lindblad operators that are going to act on that specific site, in the previously mentioned form of eq. 4 of Jaschek 2018, and "timeop" is the timeoperator constructed from "operator"
This object is simply intialized with a specific given timestep and immediately the Hamiltonian and dissipative parts are constructed.

### Time Evolution
The time evolution function of the MPS handles time evolution for a given number of steps and a given "TimeOperator" object.
You can choose whether energy and spin current through the chain are tracked. I have set these to "off" as calculating these each timestep requires some time.
You can also choose to track normalization. For DENS objects it is recommended to set this to "True", as the trace is required for calculating expectation values.
The spin profile of the chain is also calculated and stored each timestep. This is inefficient for DENS objects as it requires contracting the entire chain with itself. I have some comments regarding speed at the end of this readme.
A TEBD sweep if performed each timestep. This is mostly straightforward, the only thing of note is that for the dissipative part, the code loops through the "diss_arr" array, and applies the operator found in "timeop" at site "index".
Every 20 timestep the current timestep is printed, to keep track of progress.
After completing the time evolution for the number of desired steps, the desired tracked quantities are plotted.

### MPS Initializations
This is the second module, and contains some initialization methods. 
"Halfstate": a product state of 1/sqrt(2) * (up + down).
"LU_RD": a state with a spin profile that is a linear interpolation from completely 'up' on the left end to completely 'down' on the right end, scaled with a "scale_factor".
"Flipstate": alternating up and down spins.
"Up or Down": Everything in the up state or everything in the down state.

## Final Notes
Keep in mind that the truncation parameter should be sufficiently large. If normalization jumps to completely unphysical values (for example to 10^50) in a single timestep, chi is likely too small.
MPS type simulations require less numerical resources and are much faster than DENS type simulations.
If you are unsure if results are physical, try simulations with a small system and compare it to an exact solution.
If you are using a laptop: make sure your laptop is plugged into a power source, this drastically increases performance.
If you have any questions feel free to ask them!

### Speedups you may find useful later
I have made this code as simple as possible to use, but is therefore also not the most efficient. Should you need higher performance I have some speedups available.
First, calculating the spin profile of the entire chain requires contracting the entire MPS, which means we do a lot of calculations multiple times. I have a straightforwardly adapted routine to cut the number of redundant calculations in half.
Second, regular python code only uses a single core. It is likely your pc or laptop has multiple cores, the exact number you can check under the 'performance' of the task manager.
I have a code which parrallelizes the TEBD sweep; it distributes the application of twosite operators over all your cores to do these calculations simultanously, giving a significant speedup. Naturally, this can also be used to parrallelize other calculations should you want.
To get the most out of multiprocessing I recommend to connect your laptop to a power source. You should see a significantly higher CPU usage.
Keep in mind that using this much computer power over extended periods may affect battery life. Should you need to run a high quantity of computationally demanding tasks, you can inquire if you could be allowed to let these calculations run on a cluster.


