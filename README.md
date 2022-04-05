# Self-tuning Network Control Architecture

Code on self-tuning network control architecture.
Optimize the control architecture through greedy selection to minimize control costs.

Self-tuning architecture is based on improving the system model and the information used to select actuators.
Under full-state-feedback for a known system model, this is a run-time selection of actuators based on the current state of the system.

In the broad sense, this problem extends to the observer-based feedback problem where we base the actuator and sensor architecture at the current time step on the current estimated model of the system and work to improve the estimated model of the system over time as we accumulate more trajectory data.
