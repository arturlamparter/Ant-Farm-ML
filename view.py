"""
Dieses Modul `view` enthält alle GUI-Elemente des Projekts.

Es umfasst die Implementierungen für die grafische Darstellung mittels
Tkinter, Pygame und Matplotlib (PyPlot). Dieses Modul steuert die Visualisierung
und Benutzeroberfläche der Ameisen-Simulation.

Autor: Artur Lamparter <arturlamparter@web.de>
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pygame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import main
import model


class PyGameWindow:
    """
    GUI-Komponente für die Darstellung der Ameisenwelt mit Pygame.

    Dieses Fenster visualisiert die Positionen der Ameisen und der Nahrung auf einem Gitter.
    """

    def __init__(self, world) -> None:
        """
        Initialisiert das Pygame-Fenster mit voreingestellter Größe und Titel.
        """
        self.world = world
        self.screen = pygame.display.set_mode((self.world.screen_width, self.world.screen_height))  # Fenstergröße variabel laut config
        pygame.display.set_caption("Lernende Ameise")  # Fenstertitel setzen

    def render(self) -> None:
        """
        Zeichnet die aktuelle Welt in das Pygame-Fenster.

        Args:
            world: Ein Objekt, das die aktuellen Zustände von Ameisen und Nahrung enthält.
        """
        self.screen.fill(self.world.get_screen_color())  # Bildschirm mit Hintergrundfarbe füllen

        # Nahrung zeichnen
        for f in self.world.foods:
            fx, fy = f.get_position()  # Koordinaten abrufen
            color = f.color
            self.draw_square(color, fx, fy)  # Nahrung als grünes Quadrat

        # Ameisen zeichnen
        for a in self.world.ants:
            ax, ay = a.get_position()  # Koordinaten abrufen
            color = a.color
            self.draw_square(color, ax, ay)  # Ameise als buntes Quadrat

        pygame.display.flip()  # Zeichne den neuen Frame auf den Bildschirm

    def draw_square(self, color, x: int, y: int) -> None:             # Zeichne Quadrat in mehreren Farben
        """
        Zeichnet ein farbiges Quadrat an der gegebenen Position auf dem Grid.

        Args:
            color: Die Farbe als RGB-Tupel (z.B. model.RED).
            x (int): Die x-Koordinate im Gitter.
            y (int): Die y-Koordinate im Gitter.
        """
        rect = (
            x * self.world.grid_size,   # x-Position in Pixel
            y * self.world.grid_size,   # y-Position in Pixel
            self.world.grid_size,       # Breite
            self.world.grid_size        # Höhe
        )
        pygame.draw.rect(self.screen, color, rect)

class TkSettingsWindow(tk.Tk):
    """
    Das Hauptfenster für die Steuerung(Einstellungen) der Ameisensimulation (GUI mit Tkinter).

    Diese Klasse bildet die Benutzeroberfläche ab, mit der man:
    - die Bewegungsstrategie von Ameisen auswählen kann,
    - Geschwindigkeit, Anzahl an Ameisen und Nahrung setzen kann,
    - Lernmethoden auswählen,
    - und CSV-Daten anzeigen/speichern kann.
    """
    def __init__(self):
        super().__init__()

        # --- Allgemeine Fenstereinstellungen ---
        self.name = "Ant Settings"
        self.title(self.name)
        self.geometry("700x1000")        # Fenstergröße

        # --- Tkinter-Variablen zur Anzeige von Werten ---
        self.ants_var = tk.StringVar()  # Anzeige: Anzahl Ameisen
        self.food_var = tk.StringVar()  # Anzeige: Anzahl Futter

        # --- Callbacks ---
        self.ant_machine_learning_set = None
        self.selected_ant_set = None

        # --- OBERSTE REIHE: Strategiewahl ---
        tk.Label(self, text="Ant Strategie:", anchor="w", font=("Arial", 14)).grid(row=0, column=0, columnspan=1, padx=5)
        self.btn_random = tk.Button(self, text="Zufall")            # Zufallsbewegung (um 1 Position)
        self.btn_random.grid(row=0, column=1, columnspan=1, padx=5)

        self.btn_odor = tk.Button(self, text="Odor")                # Folgen dem Geruch(Geruchsorientiert)
        self.btn_odor.grid(row=0, column=2, columnspan=1, padx=5)

        self.btn_brain = tk.Button(self, text="Brain")              # Funktionaler Abzweig für Machine Learning
        self.btn_brain.grid(row=0, column=3, columnspan=1, padx=5)

        tk.Button(self, text="Extern").grid(row=0, column=4, columnspan=1, padx=5)           # Zuckünftige Anwendung
        tk.Button(self, text="Gesteuert").grid(row=0, column=5, columnspan=1, padx=5)      # Zuckünftige Anwendung

        # --- 2. REIHE: Geschwindigkeit / Steuerung ---
        tk.Label(self, text="Speed(FPS):", anchor="w", font=("Arial", 14)).grid(row=2, column=0, columnspan=2, padx=5, sticky="w")

        self.btn_step = tk.Button(self, text="Step")  # Pausiert/Startet die Matrix
        self.btn_step.grid(row=2, column=1, columnspan=1, padx=5)

        self.btn_pause = tk.Button(self, text=" ▶ / ⅠⅠ")                           # Pausiert/Startet die Matrix
        self.btn_pause.grid(row=2, column=2, columnspan=1, padx=5)

        self.sld_speed = tk.Scale(self, from_=1, to=60, orient="horizontal")     # Geschwindigkeit Schieberegler (Slider) , label="FPS"
        self.sld_speed.grid(row=2, column=3, columnspan=1, padx=5)

        self.btn_set_speed = tk.Button(self, text="Set >")                       # Aktiviert die eingestellte Ablaufgeschwindigkeit
        self.btn_set_speed.grid(row=2, column=4, columnspan=1, padx=5)

        self.btn_start = tk.Button(self, text=" \nStart\n ", font=("Arial", 24)) # Startet PyGame GUI, hebt Pause auf
        self.btn_start.grid(row=2, column=5, rowspan=5, padx=5)

        # --- 3. REIHE: Ant Setter ---
        tk.Label(self, text="Ants Setter:", anchor="w", font=("Arial", 14)).grid(row=4, column=0, padx=5, sticky="w")

        self.lbl_ant_var_show = tk.Label(self, textvariable=self.ants_var, anchor="center", font=("Arial", 14))
        self.lbl_ant_var_show.grid(row=4, column=1, padx=5, sticky="nsew")  # Anzahl der Ameisen unterwegs

        self.btn_ant_settings = tk.Button(self, text="EDIT")                # Einstellungsmöglichkeiten für Ant
        self.btn_ant_settings.grid(row=4, column=2, columnspan=1, padx=5)

        self.ent_ant_add = tk.Entry(self, width=7, justify='center')       # Eingabe Ameisenanzahl
        self.ent_ant_add.grid(row=4, column=3, columnspan=1, padx=5)
        self.ent_ant_add.insert(0, "0")

        self.btn_ant_add = tk.Button(self, text="Add >")                # Fügt die Ameisen in die Matrix ein
        self.btn_ant_add.grid(row=4, column=4, columnspan=1, padx=5)

        # --- 4. REIHE: Food Setter ---
        tk.Label(self, text="Food Setter:", anchor="w", font=("Arial", 14)).grid(row=6, column=0, columnspan=2, padx=5, sticky="w")

        self.lbl_food_var_show = tk.Label(self, textvariable=self.food_var, anchor="center", font=("Arial", 14))
        self.lbl_food_var_show.grid(row=6, column=1, padx=5, sticky="nsew") # Anzahl der eingestellten Futterelementen

        self.btn_food_settings = tk.Button(self, text="EDIT")               # Einstellungsmöglichkeiten für Food
        self.btn_food_settings.grid(row=6, column=2, columnspan=1, padx=5)

        self.ent_food = tk.Entry(self, width=7, justify='center') # Eingabemöglichkeit für Futterelementenanzahl
        self.ent_food.grid(row=6, column=3, columnspan=1, padx=5)
        self.ent_food.insert(0, "0")

        self.btn_set_food = tk.Button(self, text="Set >")           # Ändert die Menge der verfügbaren Futterelementen
        self.btn_set_food.grid(row=6, column=4, columnspan=1, padx=5)

        # --- 5. REIHE: Odor / Reset ---
        self.btn_show_odor = tk.Button(self, text="Show ODOR")          # Zeigt die Geruchsmatrix im PyPlot
        self.btn_show_odor.grid(row=8, column=0, columnspan=1, padx=5)

        self.show_csv_btn = tk.Button(self, text="Show CSV")  # Zeigt eine Tabellenansicht der jeweiligen CSV-Datei
        self.show_csv_btn.grid(row=8, column=1, columnspan=1, padx=5)

        self.btn_show_log = tk.Button(self, text="Show Log")
        self.btn_show_log.grid(row=8, column=2, columnspan=1, padx=5) # Zuckünftige Anwendung

        btn = tk.Button(self, text="Program Y")
        btn.grid(row=8, column=3, columnspan=1, padx=5) # Zuckünftige Anwendung

        btn = tk.Button(self, text="Program Z")
        btn.grid(row=8, column=4, columnspan=1, padx=5) # Zuckünftige Anwendung

        self.btn_reset = tk.Button(self, text="Reset")
        self.btn_reset.grid(row=8, column=5, columnspan=1, padx=5) # Zuckünftige Anwendung

        # --- 6. REIHE: Machine Learning („Brain“) ---
        tk.Label(self, text="Brain:", anchor="w", font=("Arial", 20)).grid(row=9, column=0, columnspan=1, padx=5, sticky="w")

        self.cmb_ant_machine_learning = ttk.Combobox(self, width=12)  # Ermöglicht die Auswahl der Machine Learning Methode
        self.cmb_ant_machine_learning.grid(row=9, column=1, columnspan=2, padx=5)
        self.cmb_ant_machine_learning.bind("<<ComboboxSelected>>", lambda event: self.ant_machine_learning_set())

        self.cmb_selected_ant = ttk.Combobox(self, width=12)  # Ermöglicht die Auswahl der jeweiligen Ameise
        self.cmb_selected_ant.grid(row=9, column=3, columnspan=2, padx=5)
        self.cmb_selected_ant.bind("<<ComboboxSelected>>", lambda event: self.selected_ant_set())

        # --- CSV Info / Platzhaltertext ---
        self.lbl_set_brain = tk.Label(self, font=("Arial", 14))
        self.lbl_set_brain.grid(row=10, column=0, padx=5)           # Gewählte Strategieart

        self.lbl_machine_learning = tk.Label(self, font=("Arial", 14))
        self.lbl_machine_learning.grid(row=11, column=0, padx=5)    # Gewählte Machine Learning

        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=12, column=0, padx=5) # Zuckünftige Anwendung

        frame = ttk.Frame(self)                                   # Frame für Textfeld und Scrollbar
        frame.grid(row=10, column=1, columnspan=10, rowspan=10, padx=5, pady=5, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical")       # Scrollbar
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_widget = tk.Text(frame, width=55, height=34, font=("Arial", 14), wrap="word",
        yscrollcommand=scrollbar.set, state="disabled", bg="white", relief="flat")  # Max. 10 sichtbare Zeilen
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.text_widget.yview)          # Die Scrollbar mit einem Text Widget koppeln

        # --- Letzte Reihe: CSV speichern / Programm beenden ---
        btn = tk.Button(self, text="Program")
        btn.grid(row=20, column=0, columnspan=1, padx=5)  # Zuckünftige Anwendung
        btn = tk.Button(self, text="Data") # Zuckünftige Anwendung
        btn.grid(row=20, column=1, columnspan=1, padx=5)
        btn = tk.Button(self, text="Goal") # Zuckünftige Anwendung
        btn.grid(row=20, column=2, columnspan=1, padx=5)
        btn = tk.Button(self, text="Time") # Zuckünftige Anwendung
        btn.grid(row=20, column=3, columnspan=1, padx=5)

        self.btn_save_ants = tk.Button(self, text="Save CSV") # Speichert die Machine Learning daten in der CSV
        self.btn_save_ants.grid(row=20, column=4, columnspan=1, padx=5)

        btn = tk.Button(self, text="Beenden", command=self.destroy) # App beenden
        btn.grid(row=20, column=5, columnspan=1, padx=5)

    # --- CALLBACK-VERKNÜPFUNGEN (View <- Controller) ---
    def set_btn_random_callback(self, callback):
        """Setzt den Callback für die Zufallsstrategie."""
        self.btn_random.config(command=callback)

    def set_btn_odor_callback(self, callback):
        """Setzt den Callback für die Geruchsstrategie."""
        self.btn_odor.config(command=callback)

    def set_btn_brain_callback(self, callback):
        """Setzt den Callback für die Brain/Machine-Learning-Strategie."""
        self.btn_brain.config(command=callback)

    def set_btn_step_cb(self, cb):
        self.btn_step.config(command=cb)

    def set_btn_pause_callback(self, callback):
        """Setzt den Callback für den Pause-Button."""
        self.btn_pause.config(command=callback)

    def set_btn_set_speed_callback(self, callback):
        """Setzt den Callback für den 'Set >'-Button (Geschwindigkeit)."""
        self.btn_set_speed.config(command=callback)

    def set_btn_ant_settings_cb(self, cb):
        self.btn_ant_settings.config(command=cb)

    def set_ant_add_callback(self, callback):
        """Setzt den Callback für den 'Add >'-Button (Ameisen)."""
        self.btn_ant_add.config(command=callback)

    def set_btn_food_settings_cb(self, cb):
        self.btn_food_settings.config(command=cb)

    def set_food_callback(self, callback):
        """Setzt den Callback für den 'Set >'-Button (Futter)."""
        self.btn_set_food.config(command=callback)

    def set_btn_start_callback(self, callback):
        """Setzt den Callback für den Start-Button."""
        self.btn_start.config(command=callback)

    def set_show_odor_callback(self, callback):
        """Setzt den Callback für den 'Show ODOR'-Button."""
        self.btn_show_odor.config(command=callback)

    def set_show_csv_btn_callback(self, callback):
        """Setzt den Callback für den 'Show CSV'-Button."""
        self.show_csv_btn.config(command=callback)

    def set_btn_show_log_callback(self, callback):
        """Setzt den Callback für den 'Show Log'-Button."""
        self.btn_show_log.config(command=callback)

    def set_btn_reset(self, callback):
        """Setzt den Callback für den 'Reset'-Button."""
        self.btn_reset.config(command=callback)

    def set_btn_save_ants_callback(self, callback):
        """Setzt den Callback für den 'Save CSV'-Button."""
        self.btn_save_ants.config(command=callback)

    # --- EINGABEN ABFRAGEN (View -> Controller) ---
    def get_ent_ant_add_value(self):
        """Liest die Eingabe für die Anzahl der hinzuzufügenden Ameisen."""
        return self.ent_ant_add.get()   # Ameisen generate entry

    def get_ent_set_food_value(self):
        """Liest die Eingabe für die Menge an Nahrung."""
        return self.ent_food.get()  # Hole Futter Anzahl entry

    def get_sld_speed_value(self):
        """Liest den Wert des Geschwindigkeitssliders (FPS)."""
        return self.sld_speed.get()

    def get_cmb_ant_machine_learning_value(self):
        """Liest die Auswahl aus der Machine Learning-Combobox."""
        return self.cmb_ant_machine_learning.get()

    def get_cmb_selected_ant_value(self):
        """Liest die Auswahl aus der Ameisenauswahl-Combobox."""
        return self.cmb_selected_ant.get()

    # --- ANZEIGEN UND EINTRÄGE SETZEN ---

    def update_sld_speed(self, speed):
        self.sld_speed.set(speed)

    def update_ants_label(self, ants: str) -> None:
        """Aktualisiert die Anzeige der aktuellen Ameisenanzahl."""
        self.ants_var.set(ants) # Setze self.lbl_ant_var_show Variable

    def update_food_label(self, food: str) -> None:
        """Aktualisiert die Anzeige der aktuellen Nahrungsanzahl."""
        self.food_var.set(food) # Setze self.lbl_food_var_show Variable

    def update_ent_set_food(self, food: str) -> None:
        """Setzt den Entry-Wert für Nahrung auf den gegebenen Wert."""
        self.ent_food.delete(0, tk.END)
        self.ent_food.insert(0, food)

    def update_ent_ant_add(self, ant: str) -> None:
        """Setzt den Entry-Wert für Ameisen auf den gegebenen Wert."""
        self.ent_ant_add.delete(0, tk.END)
        self.ent_ant_add.insert(0, ant)

    def update_cmb_ant_machine_learning(self, values: list[str], set_value: str, cb) -> None:
        """
        Setzt die Auswahlmöglichkeiten und den aktuell gewählten Wert der ML-Combobox.

        Args:
            values (list[str]): Liste der verfügbaren Machine-Learning-Strategien.
            set_value (str): Vorbelegung der Combobox.
        """
        self.cmb_ant_machine_learning['values'] = values
        self.cmb_ant_machine_learning.set(set_value)
        self.ant_machine_learning_set = cb

    def update_cmb_selected_ant(self, values: list[str], set_value: str, cb) -> None:
        """
        Setzt die Auswahlmöglichkeiten und den aktuell gewählten Wert der Ameisen auswahl-Combobox.

        Args:
            values (list[str]): Liste der verfügbaren Machine-Learning-Strategien.
            set_value (str): Vorbelegung der Combobox.
        """
        self.cmb_selected_ant['values'] = values
        self.cmb_selected_ant.set(set_value)
        self.selected_ant_set = cb

    def update_lbl_set_brain(self, text: str) -> None:
        """Setzt den Text des Labels für ML-Statusanzeige (Brain)."""
        self.lbl_set_brain.config(text=text)

    def update_lbl_set_machine_learning(self, text: str) -> None:
        """Setzt den Text des Labels für allgemeine Machine-Learning-Ausgabe."""
        self.lbl_machine_learning.config(text=text)

    def update_log_widget_text(self, text):
        """Setzt den Text des Text Widged für allgemeine Informationen der Ameise."""
        self.text_widget.config(state="normal") # Sonst wird nichts angezeigt !!!Wird noch geklärt!!!
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", text)
        self.text_widget.config(state="disabled")

class AntSettingsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.name = "Ant Settings"
        self.title(self.name)
        self.geometry("400x320")
        self.colors = model.COLORS
        self.chk_btn_var = tk.IntVar(value=0)  # Checkbox state (0 = off, 1 = on)

        ttk.Label(self, text="Ant Color", font=("Arial", 16)).grid(row=0, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="Random:", font=("Arial", 14)).grid(row=2, column=0, padx=10)

        self.cmb_random = ttk.Combobox(self, values=self.colors)
        self.cmb_random.grid(row=2, column=2, padx=10)

        ttk.Label(self, text="Odor:", font=("Arial", 14)).grid(row=3, column=0, padx=10)
        self.cmb_odor = ttk.Combobox(self, values=self.colors)
        self.cmb_odor.grid(row=3, column=2, padx=10)

        ttk.Label(self, text="Monte-Carlo-Methode:", font=("Arial", 14)).grid(row=4, column=0, padx=10)
        self.cmb_monte_carlo = ttk.Combobox(self, values=self.colors)
        self.cmb_monte_carlo.grid(row=4, column=2, padx=10)

        ttk.Label(self, text="Q-Learning:", font=("Arial", 14)).grid(row=5, column=0, padx=10)
        self.cmb_q_learning = ttk.Combobox(self, values=self.colors)
        self.cmb_q_learning.grid(row=5, column=2, padx=10)

        ttk.Label(self, text="", font=("Arial", 16)).grid(row=6, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="Allgemeine Einstellungen", font=("Arial", 16)).grid(row=7, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="Energie:", font=("Arial", 14)).grid(row=8, column=0, padx=10)
        self.ent = tk.Entry(self)
        self.ent.grid(row=8, column=2, padx=10)

        ttk.Label(self, text="Bei 0 entfernen:", font=("Arial", 14)).grid(row=9, column=0, padx=10)
        self.chk_btn_generationen = tk.Checkbutton(self, text="Generationen", variable=self.chk_btn_var)
        self.chk_btn_generationen.grid(row=9, column=2)

        ttk.Label(self, text="", font=("Arial", 16)).grid(row=10, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="!!! Nach änderung muss die App neugestartet werden !!!", font=("Arial", 11)).grid(row=11, column=0, columnspan=10, padx=10)
        self.btn_save = ttk.Button(self, text="Save", padding=10)
        self.btn_save.grid(row=12, column=0, padx=10)
        ttk.Button(self, text="Exit", command=self.destroy, padding=10).grid(row=12, column=2, padx=10)

    def update_settings(self, settings):
        self.cmb_random.set(settings["RANDOM_COLOR"])
        self.cmb_odor.set(settings["ODOR_COLOR"])
        self.cmb_monte_carlo.set(settings["MONTE_CARLO_COLOR"])
        self.cmb_q_learning.set(settings["Q_LEARNING_COLOR"])
        self.ent.insert(0, settings["ORKA"])
        self.chk_btn_var.set(value=settings["GENERATION"])

    def set_btn_save_cb(self, cb):
        self.btn_save.config(command=cb)

    def get_settings(self):
        settings = {"RANDOM_COLOR": self.cmb_random.get(),
                    "ODOR_COLOR": self.cmb_odor.get(),
                    "MONTE_CARLO_COLOR": self.cmb_monte_carlo.get(),
                    "Q_LEARNING_COLOR": self.cmb_q_learning.get(),
                    "ORKA": self.ent.get(),
                    "GENERATION": self.chk_btn_var.get()}
        return settings

class FoodSettingsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.name = "Food Settings"
        self.title(self.name)
        self.geometry("400x320")
        self.colors = model.COLORS
        self.chk_btn_var = tk.IntVar(value=0)  # Checkbox state (0 = off, 1 = on)

        ttk.Label(self, text="Food Color", font=("Arial", 16)).grid(row=0, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="Random Food:", font=("Arial", 14)).grid(row=2, column=0, padx=10)

        self.cmb_random = ttk.Combobox(self, values=self.colors)
        self.cmb_random.grid(row=2, column=2, padx=10)

        ttk.Label(self, text="Feste größe:", font=("Arial", 14)).grid(row=3, column=0, padx=10)
        self.cmb_fixed_size = ttk.Combobox(self, values=self.colors)
        self.cmb_fixed_size.grid(row=3, column=2, padx=10)

        ttk.Label(self, text="", font=("Arial", 16)).grid(row=6, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="Allgemeine Einstellungen", font=("Arial", 16)).grid(row=7, column=0, columnspan=10,
                                                                                  padx=10)
        ttk.Label(self, text="Food range:", font=("Arial", 14)).grid(row=8, column=0, padx=10)
        self.ent = tk.Entry(self)
        self.ent.grid(row=8, column=2, padx=10)

        ttk.Label(self, text="Random Food", font=("Arial", 14)).grid(row=9, column=0, padx=10)
        self.chk_btn_random_food = tk.Checkbutton(self, text="", variable=self.chk_btn_var)
        self.chk_btn_random_food.grid(row=9, column=2)

        ttk.Label(self, text="", font=("Arial", 16)).grid(row=10, column=0, columnspan=10, padx=10)
        ttk.Label(self, text="", font=("Arial", 11)).grid(row=11, column=0, columnspan=10, padx=10)
        self.btn_save = ttk.Button(self, text="Save", padding=10)
        self.btn_save.grid(row=12, column=0, padx=10)
        ttk.Button(self, text="Exit", command=self.destroy, padding=10).grid(row=12, column=2, padx=10)

    def update_settings(self, settings):
        self.cmb_random.set(settings["FOOD_RANDOM_COLOR"])
        self.cmb_fixed_size.set(settings["FOOD_FIXED_SIZE_COLOR"])
        self.ent.insert(0, settings["FOOD_RANGE"])
        self.chk_btn_var.set(value=settings["RANDOM_FOOD"])

    def set_btn_save_cb(self, cb):
        self.btn_save.config(command=cb)

    def get_settings(self):
        settings = {"FOOD_RANDOM_COLOR": self.cmb_random.get(),
                    "FOOD_FIXED_SIZE_COLOR": self.cmb_fixed_size.get(),
                    "FOOD_RANGE": self.ent.get(),
                    "RANDOM_FOOD": self.chk_btn_var.get()}
        return settings

class FoodOdorPyPlot:
    def __init__(self, world_array: np.ndarray) -> None:
        """
        Initialisiert das Plot-Fenster für die Geruchsausbreitung.

        Args:
            world_array (np.ndarray): 2D-Array der Geruchswerte in der Welt. (z.B. numpy-Array mit Ganzzahlen).
        """
        plt.imshow(world_array, cmap="viridis")
        plt.colorbar()
        plt.title("Geruchsausbreitung")
        plt.show()

class CSVViewer(tk.Tk):
    def __init__(self, csv_path=None):
        super().__init__()
        self.title(csv_path)
        self.geometry("600x900")  # Fenstergröße
        self.tree = None
        self.df = None

        self.setup_widgets()
        if csv_path:
            self.load_csv(csv_path)

    def setup_widgets(self):
        # Button zum Laden einer neuen Datei
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X)

        load_btn = tk.Button(btn_frame, text="CSV laden", command=self.browse_file)
        load_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # Treeview
        self.tree = ttk.Treeview(self, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=hsb.set)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV-Dateien", "*.csv")])
        if file_path:
            self.load_csv(file_path)

    def load_csv(self, path):
        try:
            self.df = pd.read_csv(path, sep=",", on_bad_lines='skip')
            self.display_dataframe()
        except Exception as e:
            model.logger.error(f"Fehler beim Laden der Datei({path}): {e}")

    def display_dataframe(self):
        # Spalten löschen
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree.delete(*self.tree.get_children())

        self.tree["columns"] = list(self.df.columns)

        for col in self.df.columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_column(_col, False))
            self.tree.column(col, width=100, anchor="center")

        for _, row in self.df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def sort_column(self, col, reverse):
        try:
            self.df[col] = pd.to_numeric(self.df[col])
        except Exception as e:
            print(f"Konnte Spalte '{col}' nicht in numerisch umwandeln: {e}")

        self.df.sort_values(by=col, ascending=not reverse, inplace=True)
        self.display_dataframe()

class TextEditor(tk.Tk):
    def __init__(self, inhalt="Inhalt", txt_path=None):
        super().__init__()
        self.inhalt = inhalt
        self.txt_path = txt_path
        self.title("Einfacher Texteditor")
        self.geometry("800x900")  # Fenstergröße

        # Frame als Container für Textfeld und Scrollbar
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True)

        # Scrollbar erstellen
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        # Textfeld mit Scrollbar verbinden
        self.textfeld = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set)
        self.textfeld.pack(side="left", fill="both", expand=True)
        self.textfeld.insert(tk.END, self.inhalt)

        scrollbar.config(command=self.textfeld.yview)

        # Menüleiste
        menuleiste = tk.Menu(self)
        datei_menu = tk.Menu(menuleiste, tearoff=0)
        datei_menu.add_command(label="Öffnen", command=self.datei_oeffnen)
        datei_menu.add_command(label="Speichern", command=self.datei_speichern)
        datei_menu.add_separator()
        datei_menu.add_command(label="Beenden", command=self.quit)
        menuleiste.add_cascade(label="Datei", menu=datei_menu)

        self.config(menu=menuleiste)

    def datei_oeffnen(self):
        pfad = filedialog.askopenfilename(
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        if pfad:
            try:
                with open(pfad, "r", encoding="utf-8") as file:
                    self.inhalt = file.read()
                self.textfeld.delete("1.0", tk.END)
                self.textfeld.insert(tk.END, self.inhalt)
            except Exception as e:
                messagebox.showerror("Fehler", f"Datei konnte nicht geöffnet werden:\n{e}")

    def datei_speichern(self):
        pfad = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        if pfad:
            try:
                inhalt = self.textfeld.get("1.0", tk.END)
                with open(pfad, "w", encoding="utf-8") as file:
                    file.write(inhalt)
            except Exception as e:
                messagebox.showerror("Fehler", f"Datei konnte nicht gespeichert werden:\n{e}")

# === Testpoint ===
if __name__ == "__main__":
    main.main()
