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

import config
import main
import model

class PyGameWindow:
    """
    GUI-Komponente für die Darstellung der Ameisenwelt mit Pygame.

    Dieses Fenster visualisiert die Positionen der Ameisen und der Nahrung auf einem Gitter.
    """

    def __init__(self) -> None:
        """
        Initialisiert das Pygame-Fenster mit voreingestellter Größe und Titel.
        """
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))  # Fenstergröße variabel laut config
        pygame.display.set_caption("Lernende Ameise")  # Fenstertitel setzen

    def render(self, world: model.World()) -> None:
        """
        Zeichnet die aktuelle Welt in das Pygame-Fenster.

        Args:
            world: Ein Objekt, das die aktuellen Zustände von Ameisen und Nahrung enthält.
        """
        self.screen.fill(config.WHITE)  # Bildschirm mit Hintergrundfarbe füllen

        # Nahrung zeichnen
        for f in world.foods:
            fx, fy = f.get_position()  # Koordinaten abrufen
            self.draw_square(config.GREEN, fx, fy)  # Nahrung als grünes Quadrat

        # Ameisen zeichnen
        for a in world.ants:
            ax, ay = a.get_position()  # Koordinaten abrufen
            self.draw_square(config.RED, ax, ay)  # Ameise als rotes Quadrat

        pygame.display.flip()  # Zeichne den neuen Frame auf den Bildschirm

    def draw_square(self, color, x: int, y: int) -> None:             # Zeichne Quadrat in mehreren Farben
        """
        Zeichnet ein farbiges Quadrat an der gegebenen Position auf dem Grid.

        Args:
            color: Die Farbe als RGB-Tupel (z.B. config.RED).
            x (int): Die x-Koordinate im Gitter.
            y (int): Die y-Koordinate im Gitter.
        """
        rect = (
            x * config.GRID_SIZE,   # x-Position in Pixel
            y * config.GRID_SIZE,   # y-Position in Pixel
            config.GRID_SIZE,       # Breite
            config.GRID_SIZE        # Höhe
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
        self.geometry("700x510")        # Fenstergröße

        # --- Tkinter-Variablen zur Anzeige von Werten ---
        self.ants_var = tk.StringVar()  # Anzeige: Anzahl Ameisen
        self.food_var = tk.StringVar()  # Anzeige: Anzahl Futter

        # ===============================================
        # OBERSTE REIHE: Strategiewahl
        # ===============================================
        tk.Label(self, text="Ant Strategie:", anchor="w", font=("Arial", 14)).grid(row=0, column=0, columnspan=1, padx=5)
        self.btn_random = tk.Button(self, text="Zufall")            # Zufallsbewegung (um 1 Position)
        self.btn_random.grid(row=0, column=1, columnspan=1, padx=5)

        self.btn_odor = tk.Button(self, text="Odor")                # Folgen dem Geruch(Geruchsorientiert)
        self.btn_odor.grid(row=0, column=2, columnspan=1, padx=5)

        self.btn_brain = tk.Button(self, text="Brain")              # Funktionaler Abzweig für Machine Learning
        self.btn_brain.grid(row=0, column=3, columnspan=1, padx=5)

        tk.Button(self, text="Extern").grid(row=0, column=4, columnspan=1, padx=5)           # Zuckünftige Anwendung
        tk.Button(self, text="Gesteuert").grid(row=0, column=5, columnspan=1, padx=5)      # Zuckünftige Anwendung

        # ===============================================
        # 2. REIHE: Geschwindigkeit / Steuerung
        # ===============================================
        tk.Label(self, text="Geschwindigkeit(FPS):", anchor="w", font=("Arial", 14)).grid(row=2, column=0, columnspan=2, padx=5, sticky="w")

        self.btn_pause = tk.Button(self, text="Pause")                           # Pausiert/Startet die Matrix
        self.btn_pause.grid(row=2, column=2, columnspan=1, padx=5)

        self.sld_speed = tk.Scale(self, from_=1, to=60, orient="horizontal")     # Geschwindigkeit Schieberegler (Slider) , label="FPS"
        self.sld_speed.grid(row=2, column=3, columnspan=1, padx=5)

        self.btn_set_speed = tk.Button(self, text="Set >")                       # Aktiviert die eingestellte Ablaufgeschwindigkeit
        self.btn_set_speed.grid(row=2, column=4, columnspan=1, padx=5)

        self.btn_start = tk.Button(self, text=" \nStart\n ", font=("Arial", 24)) # Startet PyGame GUI, hebt Pause auf
        self.btn_start.grid(row=2, column=5, rowspan=5, padx=5)

        # ===============================================
        # 3. REIHE: Ant Setter
        # ===============================================
        tk.Label(self, text="Ants Setter:", anchor="w", font=("Arial", 14)).grid(row=4, column=0, padx=5, sticky="w")

        self.lbl_ant_var_show = tk.Label(self, textvariable=self.ants_var, anchor="center", font=("Arial", 14))
        self.lbl_ant_var_show.grid(row=4, column=1, padx=5, sticky="nsew")  # Anzahl der Ameisen unterwegs

        btn = tk.Button(self, text="EDIT")                  # Einstellungsmöglichkeiten für Ant (nicht Implementiert)
        btn.grid(row=4, column=2, columnspan=1, padx=5)

        self.ent_ant_add = tk.Entry(self, width=7, justify='center')       # Eingabe Ameisenanzahl
        self.ent_ant_add.grid(row=4, column=3, columnspan=1, padx=5)
        self.ent_ant_add.insert(0, "0")

        self.btn_ant_add = tk.Button(self, text="Add >")                # Fügt die Ameisen in die Matrix ein
        self.btn_ant_add.grid(row=4, column=4, columnspan=1, padx=5)

        # ===============================================
        # 4. REIHE: Food Setter
        # ===============================================
        tk.Label(self, text="Food Setter:", anchor="w", font=("Arial", 14)).grid(row=6, column=0, columnspan=2, padx=5, sticky="w")

        self.lbl_food_var_show = tk.Label(self, textvariable=self.food_var, anchor="center", font=("Arial", 14))
        self.lbl_food_var_show.grid(row=6, column=1, padx=5, sticky="nsew") # Anzahl der eingestellten Futterelementen

        btn = tk.Button(self, text="EDIT")                  # Einstellungsmöglichkeiten für Food (nicht Implementiert)
        btn.grid(row=6, column=2, columnspan=1, padx=5)

        self.ent_food = tk.Entry(self, width=7, justify='center') # Eingabemöglichkeit für Futterelementenanzahl
        self.ent_food.grid(row=6, column=3, columnspan=1, padx=5)
        self.ent_food.insert(0, "0")

        self.btn_set_food = tk.Button(self, text="Set >")           # Ändert die Menge der verfügbaren Futterelementen
        self.btn_set_food.grid(row=6, column=4, columnspan=1, padx=5)

        # ===============================================
        # 5. REIHE: Odor / Reset
        # ===============================================
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

        # ===============================================
        # 6. REIHE: Machine Learning („Brain“)
        # ===============================================
        tk.Label(self, text="Brain:", anchor="w", font=("Arial", 20)).grid(row=9, column=0, columnspan=1, padx=5, sticky="w")

        self.cmb_ant_machine_learning = ttk.Combobox(self, width=12)  # Ermöglicht die Auswahl der Machine Learning Methode
        self.cmb_ant_machine_learning.grid(row=9, column=1, columnspan=1, padx=5)

        # self.ent_brain = tk.Entry(self, width=7, justify='center') # Eingabe möglichkeit für die Anzahl (nicht Implementiert)
        # self.ent_brain.grid(row=9, column=3, columnspan=1, padx=5)
        # self.ent_brain.insert(0, "0")

        self.btn_ant_machine_learning_set = tk.Button(self, text="Set >") # Setzt Lernmethoden
        self.btn_ant_machine_learning_set.grid(row=9, column=2, columnspan=1, padx=5)

        self.cmb_selected_ant = ttk.Combobox(self, width=12)  # Ermöglicht die Auswahl der jeweiligen Ameise
        self.cmb_selected_ant.grid(row=9, column=4, columnspan=1, padx=5)

        self.btn_selected_ant_set = tk.Button(self, text="Set >")  # Setzt Lernmethoden
        self.btn_selected_ant_set.grid(row=9, column=5, columnspan=1, padx=5)

        # ===============================================
        # CSV Info / Platzhaltertext
        # ===============================================
        self.lbl_set_brain = tk.Label(self, font=("Arial", 14))
        self.lbl_set_brain.grid(row=10, column=0, padx=5)           # Gewählte Strategieart

        self.lbl_machine_learning = tk.Label(self, font=("Arial", 14))
        self.lbl_machine_learning.grid(row=11, column=0, padx=5)    # Gewählte Machine Learning

        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=12, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=13, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=14, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=15, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=16, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=17, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=18, column=0, padx=5) # Zuckünftige Anwendung
        tk.Label(self, text=f"", font=("Arial", 14)).grid(row=19, column=0, padx=5) # Zuckünftige Anwendung

        # label = tk.Label(self, text=text, anchor="w", justify="left", font=("Arial", 14), bg="white")# Zuckünftige Anwendung
        # label.grid(row=9, column=1, columnspan=200, rowspan=200, padx=5, sticky="we" )

        frame = ttk.Frame(self)                                   # Frame für Textfeld und Scrollbar
        frame.grid(row=10, column=1, columnspan=10, rowspan=10, padx=5, pady=5, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical")       # Scrollbar
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_widget = tk.Text(frame, width=55, height=10, font=("Arial", 14), wrap="word",
        yscrollcommand=scrollbar.set, state="disabled", bg="white", relief="flat")  # Max. 10 sichtbare Zeilen
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.text_widget.yview)          # Die Scrollbar mit einem Text Widget koppeln

        # ===============================================
        # Letzte Reihe: CSV speichern / Programm beenden
        # ===============================================
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

    # ============================================================
    # CALLBACK-VERKNÜPFUNGEN (View <- Controller)
    # ============================================================

    def set_btn_random_callback(self, callback):
        """Setzt den Callback für die Zufallsstrategie."""
        self.btn_random.config(command=callback)

    def set_btn_odor_callback(self, callback):
        """Setzt den Callback für die Geruchsstrategie."""
        self.btn_odor.config(command=callback)

    def set_btn_brain_callback(self, callback):
        """Setzt den Callback für die Brain/Machine-Learning-Strategie."""
        self.btn_brain.config(command=callback)

    def set_btn_pause_callback(self, callback):
        """Setzt den Callback für den Pause-Button."""
        self.btn_pause.config(command=callback)

    def set_btn_set_speed_callback(self, callback):
        """Setzt den Callback für den 'Set >'-Button (Geschwindigkeit)."""
        self.btn_set_speed.config(command=callback)

    def set_ant_add_callback(self, callback):
        """Setzt den Callback für den 'Add >'-Button (Ameisen)."""
        self.btn_ant_add.config(command=callback)

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

    def set_btn_ant_machine_learning(self, callback):
        """Setzt den Callback für den 'Set >'-Button (ML-Ameisen)."""
        self.btn_ant_machine_learning_set.config(command=callback)

    def set_btn_selected_ant_set(self, callback):
        """Setzt den Callback für den 'Set >'-Button (Ameisenauswahl zum anzeigen)."""
        self.btn_selected_ant_set.config(command=callback)

    def set_btn_save_ants_callback(self, callback):
        """Setzt den Callback für den 'Save CSV'-Button."""
        self.btn_save_ants.config(command=callback)

    # ============================================================
    # EINGABEN ABFRAGEN (View -> Controller)
    # ============================================================

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

    # ============================================================
    # ANZEIGEN UND EINTRÄGE SETZEN
    # ============================================================

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

    def update_cmb_ant_machine_learning(self, values: list[str], set_value: str) -> None:
        """
        Setzt die Auswahlmöglichkeiten und den aktuell gewählten Wert der ML-Combobox.

        Args:
            values (list[str]): Liste der verfügbaren Machine-Learning-Strategien.
            set_value (str): Vorbelegung der Combobox.
        """
        self.cmb_ant_machine_learning['values'] = values
        self.cmb_ant_machine_learning.set(set_value)

    def update_cmb_selected_ant(self, values: list[str], set_value: str) -> None:
        """
        Setzt die Auswahlmöglichkeiten und den aktuell gewählten Wert der Ameisen auswahl-Combobox.

        Args:
            values (list[str]): Liste der verfügbaren Machine-Learning-Strategien.
            set_value (str): Vorbelegung der Combobox.
        """
        self.cmb_selected_ant['values'] = values
        self.cmb_selected_ant.set(set_value)

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
        self.geometry("300x900")  # Fenstergröße
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
            self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
            self.display_dataframe()
        except Exception as e:
            print("Fehler beim Laden der Datei:", e)

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
