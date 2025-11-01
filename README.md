# 🤖 automatica: Implementazioni di Osservatori e Filtri di Stato

Questo repository contiene implementazioni fondamentali di algoritmi di stima di stato per sistemi dinamici lineari a tempo discreto

## 📚 Concetti Implementati

Il progetto include due algoritmi chiave:

### 1. Osservatore di Luenberger
Un osservatore deterministico che stima lo stato $\mathbf{x}$ di un sistema. Il guadagno viene scelto per garantire la **convergenza asintotica** dell'errore di stima (posizionamento dei poli).

### 2. Filtro di Kalman 
Un **osservatore ottimo** che minimizza la covarianza dell'errore di stima (errore quadratico medio minimo, **MMSE**) in presenza di rumore stocastico (rumore di processo e di misura). L'algoritmo opera in un ciclo ricorsivo di **Predizione** e **Aggiornamento (Filtro)**.

---

## 💻 Installazione e Requisiti

Il progetto è sviluppato in Python 3 e richiede le librerie scientifiche standard.

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/Woodman04/automatica.git
    cd automatica
    ```

2.  **Installa le dipendenze:**
    ```bash
    pip install numpy matplotlib
    ```

---

## ▶️ Come Eseguire gli Esempi

Per eseguire gli esempi e visualizzare i risultati, utilizza l'opzione `-m` di Python per lanciare direttamente gli script di esempio.

### 1. Esecuzione dell'Osservatore di Luenberger

Lancia la simulazione e la visualizzazione della stima tramite l'Osservatore di Luenberger:

```bash
python3 -m examples.run_luenberger
python3 -m examples.run_kalman
