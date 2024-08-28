# Distributed optimization
Estensione della libreria [Disropt](https://github.com/OPT4SMART/disropt.git) 
che implementa gli algoritmi ADMM-Tracking Gradient e GIANT-ADMM.

Per ciascuno degli algoritmi implementati è fornito un launcher che lo esegue
con valori di esempio.

## Cosa fa ciascun file
- `admm_tracking_gradient.py` e `giant_admm.py` estendono la libreria disropt [Disropt](https://github.com/OPT4SMART/disropt.git) implementando rispettivamente  gli algoritmi ADMM-Tracking Gradient e GIANT-ADMM
- `admm_launcher.py`, `giant_admm_launcher.py` e `gt_launcher.py` lanciano con valori di esempio generati in modo pseudo-casuale rispettivamente gli algoritmi ADMM-Tracking Gradient, GIANT-ADMM e Gradient Tracking
- `parameters.py` definisce alcuni parametri per il fine- tuning delle simulazioni di esempio dei tre algoritmi
- `plot.py` esegue un plot dei risultati delle simulazioni di esempio dei tre algoritmi, in modo che le prestazioni possano essere confrontate
- `logistic_regression_launcher.py` utilizza tutti e tre gli algoritmi per risolvere un problema distribuito di training di un classificatore con regressione logistica. Anche il dataset viene generato casualmente attraverso il parametro opzionale `--seed` che può essere passato allo script
- `logistic_regression_parameters.py` definisce i parametri per il fine-tuning degli algoritmi usati per la regressione logistica
- `logistic_regression_plot.py` esegue il plot dei risultati di una singola simulazione
- `montecarlo_plot.py` esegue per 50 volte lo script di regressione logistica, ogni volta con semi diversi e primi, in modo da evitare sovrapposizioni.`montecarlo_run.py` utilizza i risultati di `montacarlo_plot.py` per generare dei grafici che mettano a confronti i valori medi degli errori delle simulazioni Montecarlo.

## Lavorare in un virtual environment
È consigliabile installare le dependencies in un ambiente virtuale locale al progetto, in modo da evitare conflitti con dependencies installate globalmente.

Per creare il virtual environment:
```bash
python3 -m venv env
```
Per attivarlo: 
```bash
source env/bin/activate
```
Per installare le dependencies necessarie
```bash
pip3 install -r requirements.txt
```
Per disattivarlo:
```bash
deactivate
```

## Eseguire gli script disropt
Creare le directory che ospitano i risultati:
```bash
mkdir giant_admm_results
mkdir gt_results
mkdir admm_results
mkdir logistic_regression_results
```
Eseguire lo script:
```bash
mpirun -np 8 python3 script.py
```
Dove 8 è il numero di processori e script.py è il nome dello script

## Plot dei risultati
```bash
python3 plot.py
```

## Note di teoria
Tre setup possibili per un algoritmo di ottimizzazione distribuita:

- **cost-coupled**: ogni agente i-esimo conosce solo il proprio insieme $X_i$ di constraint e la propria $f_i$ locale. Scopo del problema è far convergere la stima locale di ogni agente a uno stesso $x^*$.

- **common cost**: $f(x)$ comune a tutti gli agenti, la $x_i$ di ogni agente deve appartenere a un insieme locale di constraint $X_i$. Anche in questo caso, dobbiamo far convergere ogni stima locale a $x^*$.

- **constraint-coupled**: ogni agente conosce $f_i$ e una variabile locale $x_i$. Le variabili locali devono soddisfare sia un constraint locale $x_i \in X_i$ sia uno globale, che riguarda insieme tutte le variabili. Scopo del problema è far convergere le soluzioni locali $(x_1, ..., x_N)$ alla soluzione ottimale  $(x_1^*, ..., x_N^*)$.