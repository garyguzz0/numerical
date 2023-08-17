Most of the work that I was able to gather here is from my last year in my master's program, and is on training non-linear neural networks to learn data-assimilated solutions to the heat equation (pde)
and the Lorenz system (ode). The idea is that, in a dynamical system, we do not know the history of the trajectory. To combat this, we generate a random "history"
(i.e. we generate a random initial condition). We then introduce a "nudging" term to the differential equation, which acts to pull the nudged trajectory toward
the true trajectory. 
We also train a non-linear neural network to learn a simple case of the 2D Navier-Stokes equations (simple pipe flow). Details of this project
can be found in the "papers" folder. 
