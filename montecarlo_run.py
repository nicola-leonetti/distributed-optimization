import os
import shutil
import subprocess
import sympy
import sys

NUMBER_OF_SIMULATIONS = 50
RESULTS_DIR_BASE = "logistic_regression_results"
NUMBER_OF_AGENTS = 4

# Number of simulations completed without errors
completed_simulations = 0
i = 0
while completed_simulations < NUMBER_OF_SIMULATIONS:

    # For each simulation attempt, generate a new prime number to use as
    # a seed for the simulation's RNG
    i += 1
    seed = sympy.prime(i)
    print(f"\nStarting simulation {
          completed_simulations + 1}/{NUMBER_OF_SIMULATIONS} with seed {seed}")

    if not os.path.exists(RESULTS_DIR_BASE):
        os.makedirs(RESULTS_DIR_BASE)

    command = f"mpirun -np {NUMBER_OF_AGENTS} python3 logistic_regression_launcher.py --seed {
        seed}"
    # MPI's output is redirected to /dev/null so that it is discarded.
    # This way, the script's output is less verbose
    subprocess.run(command,
                   shell=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # If a temporary file with the seed as its name is found, the script
    # knows the simulation has failed, deletes the file and skips to the
    # next simulation
    if os.path.exists(f"{seed}"):
        print("Simulation failed")
        os.remove(f"{seed}")
        continue
    else:
        completed_simulations += 1
        print("Simulation finished")

    # Save results in a new results dir
    print("Saving the results...")
    new_results_dir = f"{RESULTS_DIR_BASE}_{seed}"
    if not os.path.exists(RESULTS_DIR_BASE):
        print(f"Base results folder {RESULTS_DIR_BASE} not found. Exiting")
        sys.exit(1)
    if os.path.exists(new_results_dir):
        shutil.rmtree(new_results_dir)
    shutil.move(RESULTS_DIR_BASE, new_results_dir)

    print("Results saved")

print("\nDone.")
