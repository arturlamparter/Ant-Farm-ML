# Ant Simulation

This project is an **Ant Simulation** with a graphical user interface.
The aim of the software is to demonstrate the principles of Machine Learning (ML) in an engaging and interactive way for students. The simulation currently includes the following algorithms:

- Random: Random movement of agents (agents move without a fixed goal).
- Odor: Agents move towards a specific odor or food source, based on a predefined procedure.
- Monte Carlo Method: Agents learn and navigate based on Q-values (Monte Carlo-based decision-making).
- Q-Learning: Agents learn through rewards and punishments (based on 4 state values).
- Perceptron: A neural network consisting of 4 perceptrons with 4 input values each.
- Neural Network: (Planned, implementation depending on project interest).

Features of the Simulation:

- Real-Time Comparison: Agents can be compared in real-time, simultaneously.
- Flexible Simulation Speed: Adjust the speed of the simulation, with options for pausing or step-by-step execution.
- Information and Log System: Track agent calculations and decisions in real-time.
- Save and Load Data: Learned states of the agents can be saved and reloaded.
- Reward and Punishment System: Agents receive rewards for finding food or are punished for aimlessly wandering.
- Generational Sorting: Successful agents reproduce, while less successful ones are removed from the simulation.
- Customizable Parameters: Variables like food amount and odor strength can be easily adjusted.
- Training the model with predefined data

The application follows the **MVC** (Model-View-Controller) architecture and serves as a learning project for Artificial Intelligence.
This structure makes it highly extensible and adaptable to future challenges.

# Ameisen-Simulation

Dieses Projekt ist eine **Ameisen-Simulation** mit einer grafischen Benutzeroberfläche. 
Ziel der Software ist es, die Funktionsweise von Maschinellem Lernen (ML) für Schüler anschaulich 
und interaktiv darzustellen. Die Simulation umfasst derzeit folgende Algorithmen:

- Random: Zufällige Bewegungen der Agenten (Agenten bewegen sich ohne festgelegtes Ziel)
- Odor: Agenten bewegen sich nach einem fest codierten Verfahren in Richtung Geruch bzw. Futter.
- Monte-Carlo-Verfahren: Agenten lernen und navigieren basierend auf Q-Werten (Monte Carlo-basierte Entscheidungsfindung)
- Q-Learning: Agenten lernen durch Belohnungen und Bestrafung (auf basis von 4 Zustandswerten)
- Perzeptron: Ein künstliches Netzwerk aus 4 Pereptronen mit jeweils 4 Eingabewerten.
- Neurales Netzwerk: (In Planung, Umsetzung abhängig vom Interesse am Projekt)

Möglichkeiten der Simulation:

- Echtzeit-Vergleich: Agenten können gleichzeitig und in Echtzeit miteinander verglichen werden.
- Flexible Ablaufgeschwindigkeit: Anpassung der Simulationsgeschwindigkeit, einschließlich der Möglichkeit, die Simulation zu pausieren oder schrittweise ablaufen zu lassen.
- Informations- und Log-System: Verfolgen von Berechnungen und Entscheidungen der Agenten in Echtzeit.
- Speichern und Laden von Daten: Gelernte Zustände der Agenten können gespeichert und wieder geladen werden.
- Belohnungs- und Bestrafungssystem: Agenten erhalten Belohnungen bei Essensfunden oder werden bestraft, wenn sie ziellos umherirren.
- Generationssortierung möglich: Erfolgreiche Agenten vermehrt sich, weniger erfolgreiche werden entfernt.
- Anpassbare Parameter: Variablen wie Essensmenge und Geruchsstärke können flexibel angepasst werden.
- Training des Models mit vordefinierten Daten

Die Anwendung folgt dem **MVC-Prinzip** (Model-View-Controller) und dient als Lernprojekt im Bereich Künstliche Intelligenz.
Dadurch ist sie flexibel erweiterbar und anpassbar für zukünftige Herausforderungen.

---

## Version

0.2.8

---

## Projektziel

Demonstration und Untersuchung von KI-gesteuerten Agenten in einer simulierten Umgebung.

---

## Technologien & Bibliotheken

- **Python 3**  
- **Pygame**: Für die grafische Darstellung und Simulation der Ameisenbewegung  
- **Tkinter**: GUI für Einstellungen und Steuerung  
- **Matplotlib**: Visualisierung von Geruchsmatrizen und anderen Daten  
- **Pandas**: Datenverarbeitung und CSV-Verwaltung  
- **NumPy**: Numerische Berechnungen  

---

## License/Lizenz

© 2025 Artur Lamparter

This repository may only be used for the purpose of review and evaluation.
Copying, modifying, distributing, or any other use of the code is prohibited without explicit written permission.

Dieses Repository darf ausschließlich zum Zwecke der Einsicht und Bewertung genutzt werden. Das Kopieren, Verändern, Weitergeben oder sonstige Verwenden des Codes ist ohne ausdrückliche schriftliche Genehmigung untersagt.


## Nutzung

Starte die Simulation mit:

```bash
python main.py


---

### Kurze **Projektstruktur**

```markdown
## Projektstruktur

- `main.py`: Einstiegspunkt der Anwendung  
- `model/`: Logik der Ameisen und Umgebung  
- `view/`: GUI-Komponenten (Tkinter + Pygame)  
- `controller/`: Steuerung der Abläufe  
- `data/`: Beispiel-Daten und CSV-Dateien  



