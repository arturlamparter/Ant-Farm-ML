"""
Dieses Modul `model` enthält alle Rechen-Elemente des Projekts.

Es umfasst die Implementierungen der gesamten Simulierten Welt. Dieses Modul stellt alle
Funktionen bereit die für die Berechnung der Ameisen-Simulation notwendig sind.

Autor: Artur Lamparter <arturlamparter@web.de>

Brain v0.7
"""
import pandas as pd
import numpy as np
import random

import config
import main

class Brain: # Hier werden Die daten als auch die Methoden für ML bereitgestellt
    """
    Die `Brain`-Klasse stellt Datenstrukturen und Algorithmen für
    einfaches Reinforcement Learning bereit, insbesondere:
    - Monte Carlo Learning
    - Q-Learning

    Q-Werte werden als Dictionary gespeichert mit:
    key   = (state, action)
    value = float
    """
    def __init__(self, name: str, ant_strategy, data=None, ant_machine_learning=None):
        self.name = name
        self._q = {}        # z.B. {(state, action): value}  (action = 'up', 'down', 'left', 'right')
        self.episode = []   # z.B. [state, action, reward] z.B. [((-1, +1, 0, -1), 'down', -0.5)]
        self.ant_strategy = ant_strategy                    # Unterschiedliche Food suche Strategien
        self.ant_machine_learning = ant_machine_learning    # Bestimte brain Methode
        self.log_collector = None
        self.alpha = 0.1
        self.gamma = 0.9

        if data is None:
            if self.ant_strategy == "brain":
                self.load_data_from_csv(self.ant_machine_learning)
        else:
            self.set_brain(data) # Wenn vorhanden setze Werte

    def __str__(self):  # Zur identifizierung des Objekts
        return self.name

    def __iter__(self):  # Ermöglich das interieren wie in einer Liste
        """Erlaubt Iteration über die Q-Werte."""
        return iter(self._q)

    def __getitem__(self, index):
        """Erlaubt Zugriff auf Q-Werte via brain[(state, action)]."""
        return self._q[index]

    def load_data_from_csv(self, ant_machine_learning):
        """Die Auswahl welche CSV geladen werden soll."""
        if ant_machine_learning == "Monte-Carlo":
            self._q = DataStorage().load_brain_data(config.MONTE_CARLO_FILE)
        elif ant_machine_learning == "Q-Learning":
            self._q = DataStorage().load_brain_data(config.Q_LEARNING_FILE)
        else:
            print("ant_machine_learning nicht Vorhanden")

    def set_brain(self, values):
            """Setzt die Q-Werte."""
            self._q = values

    def monte_carlo_calculate(self) -> None:
        """
        Berechnet neue Q-Werte auf Basis einer vollständigen Episode
        (von Endzustand rückwärts), gemäß Monte Carlo Methode.
        """
        self.log_collector.add_log_txt(f"------Berechnung der Q-Werte für Monte carlo--------\n")
        g = 0
        for state, action, reward in reversed(self.episode):
            g = reward + self.gamma * g     # zukünftige Belohnung
            old_q = self._q.get((state, action), 0)
            self._q[(state, action)] = old_q + self.alpha * (g - old_q)
            self.log_collector.add_log_txt(f"Status:{state} - Richtung:{action} - Belohnung:{reward}\n"
                                           f"Zukünftige Belohnung:{g:.2f}\n"
                                           f"Bereits vorhandener Wert für Status und Richtung:{old_q:.2f}, Ersetzt durch: {self._q[(state, action)]:.2f}\n")

    def q_learning_calculate(self, state, action, reward, next_state):
        """
        Berechnet und aktualisiert den Q-Wert nach dem Q-Learning-Verfahren.

        Args:
            state (Any): Aktueller Zustand (z.B. Tuple mit Umgebungswerten)
            action (str): Gewählte Aktion ('up', 'down', etc.)
            reward (float): Erhaltene Belohnung
            next_state (Any): Nächster Zustand nach der Aktion
        """
        self.log_collector.add_log_txt(f"------Berechnung der Q-Werte für Q-Learning--------\n")
        current_q = self._q.get((state, action), 0)
        next_q_values = [self._q.get((next_state, a), 0) for a in ['up', 'down', 'left', 'right']]
        max_next_q = max(next_q_values)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q) # Q-Learning Formel
        self._q[(state, action)] = new_q
        self.log_collector.add_log_txt(f"Status:{state} - Richtung:{action} - Zuküftige möglichkeiten:{next_q_values}\n"
                                       f"Gewählter Wert:{max_next_q:.2f}\n"
                                       f"Bereits vorhandener Wert für Status und Richtung:{current_q:.2f}, Ersetzt durch: {new_q:.2f}\n")

    def get_q_value(self, state, action):
        """
        Gibt den aktuellen Q-Wert für ein (state, action)-Paar zurück.

        Returns:
            float: Q-Wert
        """
        # print(f"Q-Wert für {state}, {action}: {self._q.get((state, action), 0):.2f}")
        return self._q.get((state, action), 0)

    def out_brain(self):
        """
        Gibt die gesamte Q-Tabelle zurück.

        Returns:
            Dict[(state, action), value]
        """
        return self._q

class Ant:
    """
    Die Ant-Klasse repräsentiert eine Ameise in der Simulation.
    Sie bewegt sich in einer Weltmatrix basierend auf verschiedenen Strategien:
    zufällig, geruchsbasiert oder mittels Reinforcement Learning (Monte Carlo, Q-Learning).
    """
    def __init__(self, world, pos_x: int, pos_y: int, brain: Brain, name: str) -> None:
        self.name = name
        self.world = world                              # Enthält alle Weltobjekte (Matrix)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.orka = 1000                                # Energie /  Schritte
        self.directions = config.DIRECTIONS             # Später Integrieren
        self.last_direction = 'XXX'                     # Alte Richtungsanzeige
        self.last_last_direction = 'XXX'                # Alte Richtungsanzeige x 2
        self.odor = 100                                 # Geruchsstärke
        self.odor_old = 0                               # Vorhergehender Geruch
        self.verkir = 0                                 # Schmerz/unwohlsein
        self.food_found = 0                             # Foods gefunden
        self.period = 0                                 # Schritte gegangen
        if brain and isinstance(brain, Brain):          # Wenn Brain korrekt übergeben wird
            self.brain = brain # Enthält das Objekt     # Setzen
        else:
            print("Fehler beim Gehirn übergabe")        # Fehler
        self.log_collector = LogCollector(self.name, self.brain.ant_strategy, self.brain.ant_machine_learning)
        self.brain.log_collector = self.log_collector

    def get_position(self):                          # Ant Position
        """
        Gibt die aktuelle (x, y)-Position der Ameise in der Welt zurück.

        Returns:
            Tuple[int, int]: Die aktuelle Koordinate der Ameise (pos_x, pos_y).
        """
        return self.pos_x, self.pos_y

    def get_position_turned(self):                      # Position gedreht
        """
        Gibt die (y, x)-Position der Ameise zurück.

        Diese Darstellung ist für den Zugriff auf zweidimensionale Arrays im Format [row][column] nützlich,
        z.B. beim Zugriff auf NumPy-Matrizen oder World-Arrays.

        Returns:
            Tuple[int, int]: Die vertauschte Koordinate (pos_y, pos_x).
        """
        return self.pos_y, self.pos_x

    def set_pos(self, pos_x: int, pos_y: int) -> None:        # Position setzen auserhalb des Quadrat nicht erlaubt
        """
        Setzt die Position der Ameise. Falls die neue Position außerhalb der Welt liegt,
        wird sie auf die gegenüberliegende Seite gesetzt (Wrap-Around).

        Args:
            pos_x (int): Neue X-Position.
            pos_y (int): Neue Y-Position.
        """
        if pos_x < 0:                                   # Bim Bereich verlassen
            self.pos_x = 99                             # Entgegengesetzter Seite rein
        elif pos_x >= config.GRID_WIDTH:
            self.pos_x = 0
        else:
            self.pos_x = pos_x

        if pos_y < 0:
            self.pos_y = 99
        elif pos_y >= config.GRID_WIDTH:
            self.pos_y = 0
        else:
            self.pos_y = pos_y

    def move_direction(self, direction: str) -> None: # Bewegung
        """
        Bewegt die Ameise in die angegebene Richtung.

        Args:
            direction (str): Eine der vier Richtungen: 'up', 'down', 'left', 'right'.
        """
        if direction == 'up':
            self.set_pos(self.pos_x, self.pos_y - 1)
        elif direction == 'down':
            self.set_pos(self.pos_x, self.pos_y + 1)
        elif direction == 'left':
            self.set_pos(self.pos_x - 1, self.pos_y)
        elif direction == 'right':
            self.set_pos(self.pos_x + 1, self.pos_y)
        else:
            pass
            # print(f"Keine richtung vorhanden.({direction})")

    def move(self) -> None: # Hier wird die Ameise bewegt mit pos_x und pos_y
        """
        Führt eine Bewegung der Ameise aus, abhängig von der aktuell
        gewählten Strategie (random, odor, brain).
        """
        if self.brain.ant_strategy == "random":
            self.move_random()
        elif self.brain.ant_strategy == "odor":
            self.move_odor()
        elif self.brain.ant_strategy == "brain":
            self.move_brain()

    def move_brain(self) -> None: # Auswahl ML Verfahren
        """
        Führt eine Bewegung anhand der im Brain definierten Lernmethode aus
        (Monte-Carlo oder Q-Learning).
        """
        if self.brain.ant_machine_learning == "Monte-Carlo":
            self.move_brain_monte_carlo()
        elif self.brain.ant_machine_learning == "Q-Learning":
            self.move_brain_q_learning()

    def calculate_state(self): # Geruchsunterschiede als Zustand
        """
        Berechnet den aktuellen Zustand basierend auf Geruchsunterschieden in den vier Richtungen.

        Returns:
            tuple[int, int, int, int]: Geruchsunterschiede (oben, unten, links, rechts),
            jeweils im Bereich [-1, 0, 1].
        """
        odor_up = int(max(-1, min(1, self.world.get_odor(self.pos_x, self.pos_y - 1) - self.odor)))     # Hole Geruch Oben
        odor_down = int(max(-1, min(1, self.world.get_odor(self.pos_x, self.pos_y + 1) - self.odor)))   # Hole Geruch Unten
        odor_left = int(max(-1, min(1, self.world.get_odor(self.pos_x - 1, self.pos_y) - self.odor)))   # Hole Geruch Links
        odor_right = int(max(-1, min(1, self.world.get_odor(self.pos_x + 1, self.pos_y) - self.odor)))  # Hole Geruch Rechts
        return odor_up, odor_down, odor_left, odor_right

    def move_brain_q_learning(self) -> None:
        """
        Führt eine Bewegung mithilfe des Q-Learning-Verfahrens durch.
        Berechnet Belohnung und aktualisiert das Q-Table im Brain.
        """
        state = self.calculate_state()                                      # Aktueller Status
        q_value = [round(float(self.brain.get_q_value(state, d)), 1) for d in self.directions] # Q-Werte für alle Richtungen
        q_max = max(q_value)
        new_directions = [d for q, d in zip(q_value, self.directions) if q == q_max]
        action = random.choice(new_directions)
        self.log_collector.add_log_txt(f"Positionsgeruch:{self.odor}, Berechneter State:{state}, Anfrage Q-Value:{q_value}, Q-Max:{q_max},\n"
            f" Mögliche Richtungen:{new_directions}, Bestimmt:{action}, letzte aktion:{self.last_direction}\n")
        # Rückwärtsgehen vermeiden
        opposites = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        if opposites.get(action) == self.last_direction:
            directions = self.directions.copy()
            directions.remove(action)
            action = random.choice(directions)
            self.log_collector.add_log_txt(f"Ereignis Zurück gehen eingetrofen\n"
                f"Vorhergehende Richtungswahl: {self.last_direction}\n"
                f"Zurück gehen nicht erlaubt. Neue Richtungswahl: {action}\n")
        if random.random() < 0.1:                                           # Exploration: zufällige Richtung wählen
            action = random.choice(self.directions)                         # self.direction = ['up', 'down', 'left', 'right']
            self.log_collector.add_log_txt(f"Zufällig gehen eingetroffen:{action}\n")
        self.move_direction(action)                                         # Bewegung ausführen
        self.last_direction = action
        old_odor = self.odor                                                # Alten Geruch merken
        self.odor = self.world.get_odor(self.pos_x, self.pos_y)             # Neuen Geruch holen
        next_state = self.calculate_state()                                 # Zucküftiger Status
        reward = float(round(max(-1, min(1, self.odor - old_odor)) - 0.2, 2)) # Belohnung berechnen
        for food in self.world.foods: # Futter gefunden
            if food.get_position() == self.get_position():
                reward += 5
                self.log_collector.add_log_txt(f"Essen gefunden\n")
        self.brain.q_learning_calculate(state, action, reward, next_state) # Q-Learning Update
        self.log_collector.add_log_txt(f"!!!Bewegung!!!\n"
            f"Neuer Positionsgeruch:{self.odor}, Berechneter Nexter State:{next_state}\n"
            f" Anfrage Q-Value:{q_value}, Belohnung:{reward} neuer Geruch:{self.odor}\n")
        self.log_collector.add_new_period()

    def move_brain_monte_carlo(self) -> None: # print(f"{}")
        """
        Führt eine Bewegung mithilfe des Monte-Carlo-Verfahrens durch.

        Die Methode bewertet mögliche Bewegungsrichtungen anhand vorhandener Q-Werte,
        wählt eine Richtung aus, führt die Bewegung aus und speichert
        den Übergang (state, action, reward) in der Episode.
        Falls Futter gefunden wird, wird die Monte-Carlo-Rückpropagierung ausgelöst.
        """
        state = self.calculate_state()                                                  # Aktueller Status
        q_value = [round(float(self.brain.get_q_value(state, d)), 1) for d in self.directions] # Liste von 4 Q Werten
        q_max = max(q_value)                                                            # Höchster wert gewinnt
        new_directions = [d for q, d in zip(q_value, self.directions) if q == q_max]    # Vier Richtungen durchgehen
        action = random.choice(new_directions)                                          # Richtung Höchster wert merken
        self.log_collector.add_log_txt(
            f"Position: X: {self.pos_x}, Y: {self.pos_y}, Energie: {self.orka}, Futtergefunden: {self.food_found}\n"
            f"------------------------------------------------------------------------------------------\n"
            f"Geruchswahrnehmung Position: {self.odor}\n"
            f"Oben: {state[0]}      Unten: {state[1]}       Links: {state[2]}       Rechts: {state[3]}\n"
            f"Gefundene Erfahrungen im Brain:\n"
            f"Oben: {q_value[0]}   Unten: {q_value[1]}     Links: {q_value[2]}     Rechts: {q_value[3]}\n"
            f"Berechnete Richtungen: {new_directions} Gewählte Richtung: {action}\n"
            f"Vorhergehende Richtungswahl: {self.last_direction} \n")
        opposites = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        if opposites.get(action) == self.last_direction:                # zurück gehen verboten
            directions = self.directions.copy()                         # Richtungen kopieren
            del directions[directions.index(action)]                    # Zurück löschen
            action = random.choice(directions)                          # Richtung Zufällig ohne Zurück
            self.log_collector.add_log_txt(
                f"Ereignis Zurück gehen eingetrofen\n"
                f"Vorhergehende Richtungswahl: {self.last_direction}\n"
                f"Zurück gehen nicht erlaubt. Neue Richtungswahl: {action}\n")
        if random.random() < 0.1:                                       # Exploration: zufällige Richtung wählen
            action = random.choice(self.directions)                     # self.direction = ['up', 'down', 'left', 'right']
            self.log_collector.add_log_txt(
                f"Ereignis Zufällig gehen eingetrofen\n"
                f"Vorhergehende Richtungswahl: {self.last_direction}, Neue Richtungswahl: {action}\n")
        self.move_direction(action)                                         # Bewege dich
        self.last_direction = action
        reward =  float(round(max(-1, min(1,self.world.get_odor(self.pos_x, self.pos_y) - self.odor)) - 0.2, 2)) # Belonung Geruch + oder -, strafe,
        self.log_collector.add_log_txt(f"Bewegung nach: {action}, Belohnung berechnet: {reward}\n")
        self.odor = self.world.get_odor(self.pos_x, self.pos_y)             # Hole Geruch
        for food in self.world.foods:                                       # Foods durchsuchen
            if food.get_position() == self.get_position():                  # Wenn Futter gefunden
                reward += 10                                                 # Belohnung
                self.brain.episode.append((state, action, reward))          # Schreibe den Datensatz in episode
                self.log_collector.add_log_txt(f"!!!!!!Essen gefunden!!!!!!\n"
                                               f"Belohnung: {reward}, Merke in Episode:{(state, action, reward)}\n")
                self.brain.monte_carlo_calculate()                          # Schreibe in Q Datensatz
                return                                                      # Datensatz schon geschrieben, Raus
        self.brain.episode.append((state, action, reward))                  # Schreibe den Datensatz in episode
        self.log_collector.add_log_txt(f"Merke in Episode:{(state, action, reward)}\n")
        self.log_collector.add_new_period()

    def move_odor(self) -> None:  # Kombination aus Zufall und "Geruch folgen" Strategie
        """
        Führt eine Bewegung anhand der lokalen Geruchsunterschiede aus.

        Die Ameise vergleicht den Geruch in den vier Nachbarfeldern und
        bewegt sich in die Richtung mit dem stärksten Geruch.
        Bei Gleichstand wird zufällig aus den besten Richtungen gewählt.
        """
        self.odor = self.world.get_odor(self.pos_x, self.pos_y)
        state = self.calculate_state()  # Aktueller Status
        max_value = max(state)  # Höchster wert gewinnt
        new_directions = [d for q, d in zip(state, self.directions) if q == max_value]  # Vier Richtungen durchgehen
        action = random.choice(new_directions)  # Richtung Höchster wert merken
        self.log_collector.add_log_txt(
            f"Ameise:{self.name}, Strategie: {self.brain.ant_strategy}, Lernmethode:{self.brain.ant_machine_learning}\n"
            f"Position: X:{self.pos_x}, Y:{self.pos_y}, Energie:{self.orka}, Futtergefunden:{self.food_found}\n"
            f"------------------------------------------------------------------------------------------\n"
            f"Geruchswahrnehmung Position: {self.odor}\n"
            f"Oben:{state[0]} Unten:{state[1]} Links:{state[2]} Rechts:{state[3]}\n"
            f"Berechnete Richtungen:{new_directions} Gewählte Richtung:{action}\n"
            f"------------------------------------------------------------------------------------------\n")
        self.move_direction(action)
        self.log_collector.add_new_period()

    def move_random(self) -> None: # Zufällig immer eine Position springen
        """
        Bewegt die Ameise zufällig in eine der vier Richtungen:
        'up', 'down', 'left' oder 'right'.

        Diese Methode ignoriert Geruch, Strategie oder andere Faktoren.
        """
        self.log_collector.add_log_txt(f"Position: X:{self.pos_x}, Y:{self.pos_y}, Energie:{self.orka}, Futtergefunden:{self.food_found}\n")
        self.move_direction(random.choice(self.directions))
        self.log_collector.add_new_period()

class Ants:
    """
    Eine Sammlung von Ameisen (Ant-Objekten), die auf einer gemeinsamen Welt (World) operieren.
    Diese Klasse verwaltet die Erzeugung, Iteration und Verwaltung der Ameisen.
    """
    def __init__(self, world) -> None:
        """
        Initialisiert die Ants-Sammlung mit einer Referenz auf die Welt.

        :param world: Das World-Objekt, in dem sich die Ameisen bewegen.
        """
        self.world = world
        self._ants = []

    def __iter__(self):
        """Ermöglicht die Iteration über alle Ant-Objekte."""
        return iter(self._ants)

    def __delitem__(self, index) -> None:
        """Löscht eine Ameise anhand ihres Index."""
        del self._ants[index]  # del liste[1]  löscht das Element über Index

    def __len__(self) -> int:
        """Gibt die Anzahl der Ameisen zurück."""
        return len(self._ants)

    def index(self, item: Ant) -> int:
        """
        Gibt den Index einer bestimmten Ameise zurück.

        :param item: Die zu suchende Ameise.
        :return: Der Index der Ameise in der internen Liste.
        """
        return self._ants.index(item)

    def clear(self):  # Leert die Liste
        """
        Leert die gesamte Liste der Ameisen.
        """
        self._ants.clear()

    def generate_ants(self, crowd: int, ant_strategy: str, ant_machine_learning: str = None) -> None:
        """
        Erzeugt eine Anzahl von Ameisen mit gegebener Strategie (und ggf. ML-Verfahren).

        :param crowd: Anzahl der zu erzeugenden Ameisen.
        :param ant_strategy: Bewegungsstrategie ('random', 'odor', 'brain').
        :param ant_machine_learning: Optional, bei Strategie 'brain': ML-Methode ('Monte-Carlo', 'Q-Learning').
        """
        for i in range(crowd):
            pos_x = random.randint(2, config.GRID_WIDTH - 2)
            pos_y = random.randint(2, config.GRID_HEIGHT - 2)
            if ant_strategy == "brain":
                brain = Brain(name="BrainXX", ant_strategy=ant_strategy, ant_machine_learning=ant_machine_learning)
                self._ants.append(Ant(self.world, pos_x, pos_y, brain, name=str(len(self.world.ants) + 1).zfill(3)))
            else:
                brain = Brain(name="BrainXX", ant_strategy=ant_strategy, ant_machine_learning="Keine")
                self._ants.append(Ant(self.world, pos_x, pos_y, brain, name=str(len(self.world.ants) + 1).zfill(3)))

    def show_ants(self):
        """
        Gibt alle gespeicherten Ameisen als Liste zurück.

        :return: Liste von Ant-Instanzen.
        """
        return [x for x in self._ants]

class Food:
    """
    Repräsentiert eine Futterquelle mit Position, Kalorienwert und Namen.
    """
    def __init__(self, pos_x: int, pos_y: int, calories: int = 10, name: str = "Zucker") -> None:
        """
        Initialisiert das Futterobjekt.

        :param pos_x: X-Position des Futters.
        :param pos_y: Y-Position des Futters.
        :param calories: Kalorienwert, den das Futter bietet.
        :param name: Name des Futters.
        """
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.calories = calories

    def get_position(self):                 # Gebe die Position vom Futter zurück
        """
        Gibt die aktuelle Position des Futters zurück.

        :return: Tupel (pos_x, pos_y).
        """
        return self.pos_x, self.pos_y

    def set_new_position(self) -> None:             # Wenn das Futter gefunden wurde, neu Positionieren
        """
        Setzt eine neue zufällige Position für das Futter innerhalb der erlaubten Weltgrenzen.
        """
        self.pos_x = random.randint(2, config.GRID_WIDTH - 2)
        self.pos_y = random.randint(2, config.GRID_HEIGHT - 2)

class Foods:
    """
    Verwaltung einer Sammlung von Food-Objekten in der Welt.
    Ermöglicht Erzeugen, Iterieren, Löschen und Anzeigen von Futter.
    """
    def __init__(self) -> None:
        """
        Initialisiert die Foods-Sammlung.
        """
        self._foods = []
        self.set_food = 0

    def __iter__(self):
        """
        Ermöglicht Iteration über die Foods-Liste.
        """
        return iter(self._foods)

    def __delitem__(self, index) -> None:
        """
        Löscht Food-Objekt an gegebener Position.
        """
        # print(f"Futter ({self._foods[index].name}) wurde gegessen. +10 Energie")
        del self._foods[index]  # del liste[1]  löscht das Element über Index

    def __len__(self) -> int:
        """
        Gibt die Anzahl der Foods zurück.
        """
        return len(self._foods)

    def index(self, item: Food):
        """
        Gibt den Index eines Food-Objekts zurück.
        """
        return self._foods.index(item)

    def clear(self):  # Leert die Liste
        """
        Leert die gesamte Liste der Foods.
        """
        self._foods.clear()

    def generate_food(self, crowd: int = 1):
        """
        Erzeugt mehrere Food-Objekte an zufälligen Positionen.
        """
        for _ in range(crowd): #random.randint(5, 100)
            self._foods.append(Food(random.randint(2, config.GRID_WIDTH - 2), random.randint(2, config.GRID_HEIGHT - 2), 120))
        self.set_food = len(self._foods)

    def show_foods(self):
        """
        Gibt eine Liste aller Food-Objekte zurück.
        """
        return [x for x in self._foods]

class World:
    """
    Repräsentiert die Welt mit Ameisen, Futter und dem Geruchsfeld.
    Verwalten der Geruchsausbreitung und Zugriff auf Ameisen und Futter.
    """
    def __init__(self):
        """
        Initialisiert die Welt, Ameisen, Futter und das Geruchsfeld.
        """
        self.world_array = []
        self.clock_tick = 1
        self.world_pause = False
        self.ants = Ants(self)
        self.foods = Foods()
        self.update_odor_world()

    def update_odor_world(self):
        """
        Aktualisiert das Geruchsfeld basierend auf aktuellem Futter.
        """
        self.world_array = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=int) # Erzeuge Feld
        yy, xx = np.indices(self.world_array.shape)  # erstellt zwei Arrays die die Koordinaten (Indices) aller Zellen im world_array enthalten
        for food in self.foods.show_foods():
            self.calculate_odor_field(food, yy, xx)             # Berechne Geruchsausbreitung
        self.world_array[0, :] = 0                              # Setze Oberste Reihe auf Null
        self.world_array[-1, :] = 0                             # Setze Unterste Reihe auf Null
        self.world_array[:, 0] = 0                              # Setze Erste Spalte auf Null
        self.world_array[:, -1] = 0                             # Setze Letzte Spalte auf Null
        # print(self.world_array[95:100, 95:100])

    def calculate_odor_field(self, food, yy, xx):
        """
        Berechnet Geruchsausbreitung vom Futter ausgehend.
        """
        start_x, start_y = food.get_position()                  # Futter Position
        dist = np.sqrt((yy - start_y) ** 2 + (xx - start_x) ** 2)  # Euklidische Distanz zum Startpunkt berechnen
        # Werte nach Entfernung setzen: runde abwärts, invertiere zur Höhe
        value_array = np.clip(food.calories - np.floor(dist).astype(int), 0, food.calories)
        self.world_array = np.maximum(self.world_array, value_array) # Mit dem Zielarray kombinieren
        # print(self.world_array[start_y - 5:start_y + 6, start_x - 5:start_x + 6])

    def get_odor(self, x: int, y: int) -> int:
        """
        Gibt die Geruchsstärke an Position (x,y) zurück.
        """
        if 0 <= x < config.GRID_WIDTH and 0 <= y < config.GRID_HEIGHT:
            return self.world_array[y, x]   # Normalrückgabe
        return 0            # Geruch auserhalb des Array auf 0 Setzen, fehler ausschließen
        # print(f"Auserhalb des bereichs: Pos: {x}, Pos: {y}")

class LogCollector:
    """
    Sammelt und verwaltet logische Zeitabschnitte ("Perioden") von Textlogs für eine Ameisen-Simulation.
    Diese Klasse dient zur strukturierten Aufbereitung von internen Daten und Ereignissen,
    um den Ablauf und Entscheidungen der Ameise besser nachvollziehen zu können.
    Die gesammelten Logs werden als Text an einen Controller übergeben und können
    im GUI angezeigt werden.
    Attributes:
        name (str): Name der Ameise.
        ant_strategy (str): Bezeichnung der verwendeten Strategie.
        ant_machine_learning (str): Beschreibung der Lernmethode.
        _periods (List[str]): Liste der Textabschnitte, jeweils eine Periode.
        period (int): Index der aktuellen Periode.
    """

    SEPARATOR = "-" * 90 + "\n"
    PERIOD_SEPARATOR = "X" * 90 + "\n"

    def __init__(self, name, ant_strategy, ant_machine_learning):
        """
        Initialisiert den LogCollector mit den Angaben zur Ameise, Strategie und Lernmethode.
        Startet die Log-Sammlung mit einer initialen Periode.
        Args:
            name (str): Name der Ameise.
            ant_strategy (str): Verwendete Strategie der Ameise.
            ant_machine_learning (str): Lernmethode der Ameise.
        """
        self.name = name
        self.ant_strategy =  ant_strategy
        self.ant_machine_learning = ant_machine_learning
        self._periods = [f"Ameise:{name}, Strategie: {ant_strategy}, Lernmethode:{ant_machine_learning}\n"]
        self.period = 0
        self.update_log_text_widget = None

    def add_log_txt(self, text):
        """
        Fügt der aktuellen Periode einen neuen Log-Textabschnitt hinzu.
        Der Text wird mit einem Trenner getrennt angehängt.
        Args:
            text (str): Der hinzuzufügende Logtext.
        """
        self._periods[self.period] += self.SEPARATOR + text

    def add_new_period(self):
        """
        Schließt die aktuelle Periode ab und startet eine neue.
        Dabei wird der Periodenzähler erhöht und eine neue Textperiode mit
        Kopfzeile für die Ameise, Strategie und Lernmethode angelegt.
        """
        self.update_log_text_widget()
        self.period += 1
        self._periods.append(self.PERIOD_SEPARATOR + f"Ant:{self.name}, Strategie: {self.ant_strategy}, Lernmethode:{self.ant_machine_learning}, Step: {self.period}\n")

    def get_formatted_info(self):
        """
        Gibt den Text der aktuellen Periode zurück.
        Returns:
            str: Logtext der aktuellen Periode.
        """
        return self._periods[self.period-1]

    def get_all_periods(self):
        """
        Gibt die gesammelten Logtexte aller Perioden zusammenhängend zurück.
        Returns:
            str: Kombinierter Logtext aller Perioden.
        """
        return "".join(self._periods)

    def update_log_collector_callback(self, callback):
        """
        Callback vom Controller um das Text Widget zu aktualisieren
        """
        self.update_log_text_widget = callback

class DataStorage:
    """
    Klasse zur Speicherung und zum Laden von Q-Learning-Daten in CSV-Dateien.

    Die Q-Daten werden als Dictionary mit Schlüsseln aus (state, action)-Tupeln
    und float-Werten für die Q-Werte verwaltet.
    """
    def __init__(self):
        pass # Aktuell keine Initialisierung nötig

    @staticmethod
    def save_q_to_csv(q, file: str) -> None:
        """
        Speichert das Q-Dictionary in eine CSV-Datei.

        Args:
            q (Dict[(Tuple[int, int, int, int], str), float]): Q-Learning Werte.
            file (str): Pfad zur Ausgabedatei.
        """
        array = pd.DataFrame(columns=["state", "action", "value"]) # Beispiel: ["-1:0:-1:0", "up", -0.25],
        for (state, action), value in q.items(): # Fülle Panda DataFrame mit lernerfolgen der ML
            array.loc[len(array)] = (f"{state[0]}:{state[1]}:{state[2]}:{state[3]}", action, f"{value:.2f}")
            # print(f"Q-Wert für {state}, {action}: {value:.2f}")
            # print(file)
        print(f"Speichere {file}")
        array.to_csv(file, index=False) # Speicher Array in CSV

    def load_brain_data(self, file: str):
        """
        Lädt die Q-Werte aus einer CSV-Datei oder initialisiert Standardwerte,
        falls die Datei leer oder nicht vorhanden ist.

        Args:
            file (str): Pfad zur CSV-Datei.

        Returns:
            Dict[(Tuple[int, int, int, int], str), float]: Q-Werte.
        """
        data = self.load_data_from_csv_file(file)
        if data is None or data.empty:
            data = pd.DataFrame(columns=["state", "action", "value"])
            data.loc[len(data)] = ("0:0:0:0", "up", 0)
            data.loc[len(data)] = ("0:0:0:0", 'down', 0)
            data.loc[len(data)] = ("0:0:0:0", 'left', 0)
            data.loc[len(data)] = ("0:0:0:0", 'right', 0)
        q = {}
        for row in data.itertuples():
            state = tuple([int(x) for x in row.state.split(":")])
            # print(f"{state}, {row[2]},  {row[3]}")
            q[(state, row[2])] = row[3]
        return q

    @staticmethod
    def load_data_from_csv_file(file):
        """
        Liest eine CSV-Datei als pandas DataFrame ein.

        Args:
            file (str): Pfad zur CSV-Datei.

        Returns:
            Optional[pd.DataFrame]: Eingelesenes DataFrame oder None bei Fehler.
        """
        try:
            print(f"Lade {file}")
            return pd.read_csv(file, index_col=False)  # Liest die CSV-Datei als Panda DataFrame ein
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Fehler beim Lesen der CSV-Datei: {e}")
            return pd.DataFrame()  # Gibt ein leeres DataFrame zurück als "Fallback"

# === Testpoint ===
if __name__ == "__main__":
    main.main()