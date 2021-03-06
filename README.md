# Self-tuning Network Control Architecture: Results 1

We compare the control costs and performance from run-time greedy actuator selection and full-state feedback with current state information to a random set of design-time actuators and the corresponding fixed full-state feedback.

Dynamics: 50 node randomly generated well-connected ER network (open-loop unstable with eigenvalue magnitude 1.05)

In the broad sense, this problem extends to the observer-based feedback problem where we base the actuator and sensor architecture at the current time step on the current estimated model of the system and work to improve the estimated model of the system over time as we accumulate more trajectory data.

## Organization of files and branches
### main_dev
- Test branch for code
### Results 1
- Comparison of design-time vs run-time greedy control architecture - Effect of system information
- Check File [here](Test 1.ipynb) or ../Test 1.ipynb
### Results 2
- Comparison of design-time vs run-time greedy control architecture - targeted disturbances
- Check File [here](Test 2.ipynb) or ../Test 2.ipynb