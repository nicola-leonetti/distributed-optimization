# Distributed optimization
Estensione della libreria [disropt](https://github.com/OPT4SMART/disropt.git) 
che implementa gli algoritmi ADMM-Tracking Gradient e GIANT-ADMM.

Per ciascuno degli algoritmi implementati è fornito una launcher che lo esegue
con valori di esempio.

## Lavorare in un virtual environment
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