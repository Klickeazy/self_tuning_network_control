# Self-tuning Network Control Architecture - Submitted for CDC'22

We compare the control costs and performance from run-time greedy actuator selection and full-state feedback with current state information to a random set of design-time actuators and the corresponding fixed full-state feedback.

Dynamics: 50 node randomly generated well-connected ER network (open-loop unstable with eigenvalue magnitude 1.05)

## Python
Use [here](work_env.yaml) to install a conda environment with all relevant packages for the code

## File Organization
- [here](Test_Notebook.ipynb) is the jupyter notebook of the numerical analysis
- [here](functionfile_model.py) contains the functions used in the numerical analysis



<!-- 
### FileDev Information & Organization of files and branches
#### main_dev
- Test branch for code
#### Results 1
- Comparison of design-time vs run-time greedy control architecture - Effect of system information
- Check File [here](Test 1.ipynb) or ../Test 1.ipynb
#### Results 2
- Comparison of design-time vs run-time greedy control architecture - targeted disturbances
- Check File [here](Test 2.ipynb) or ../Test 2.ipynb
-->