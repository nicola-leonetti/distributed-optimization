import os
import subprocess
import shutil

from sympy import prime

NUMBER_OF_ITERATIONS = 50
RESULTS_DIR_BASE = "logistic_regression_results"

# Run many times, each with different seeds
i = 1
while i < NUMBER_OF_ITERATIONS:
    i += 1
    seed = prime(i)

    if not os.path.exists(RESULTS_DIR_BASE):
        os.makedirs(RESULTS_DIR_BASE)

    command = f"mpirun -np 4 python3 logistic_regression_launcher.py --seed {
        seed}"
    try:
        subprocess.run(command,
                       shell=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except:
        print("Error while running the script. Ignoring this seed...")

    # Save results in a new results dir
    new_results_dir = f"{RESULTS_DIR_BASE}_{seed}"
    if not os.path.exists(RESULTS_DIR_BASE):
        print(f"Base results folder {RESULTS_DIR_BASE} not found. Exiting")
        exit

    shutil.move(RESULTS_DIR_BASE, new_results_dir)

    print(f"Completed simulation {i}/{NUMBER_OF_ITERATIONS} with seed {seed}")
