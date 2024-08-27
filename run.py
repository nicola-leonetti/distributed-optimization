import os
import subprocess
import shutil

from sympy import prime

NUMBER_OF_ITERATIONS = 50
RESULTS_DIR_BASE = "logistic_regression_results"

# Esegui 50 volte con seed diversi
for i in range(1, NUMBER_OF_ITERATIONS + 1):
    seed = prime(i)

    if not os.path.exists(RESULTS_DIR_BASE):
        os.makedirs(RESULTS_DIR_BASE)

    command = f"mpirun -np 4 python3 logistic_regression_launcher.py --seed {
        seed}"
    subprocess.run(command, shell=True)

    # Rinomina la cartella dei risultati
    new_results_dir = f"{RESULTS_DIR_BASE}_{seed}"
    if os.path.exists(RESULTS_DIR_BASE):
        shutil.move(RESULTS_DIR_BASE, new_results_dir)
    else:
        print(f"Cartella {
              RESULTS_DIR_BASE} non trovata. Assicurati che il processo generi questa cartella.")

    print(f"Completata esecuzione {
          seed+1}/{NUMBER_OF_ITERATIONS} con seed {seed}. Risultati salvati in {new_results_dir}")

print("Tutte le esecuzioni sono state completate.")
