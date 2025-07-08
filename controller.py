"""
Dieses Modul `controller` enthält alle Steuerungs-Elemente des Projekts.

Es umfasst die Implementierungen für die Steuerung allen Details. Dieses Modul steuert das gesamte Projekt.
Benutzeroberfläche der Einstellungen und die Darstellung der Ameisen-Simulation.

Autor: Artur Lamparter <arturlamparter@web.de>
"""
import threading

import pygame

import config
import main
import view
import model

class PyGameController:
    """
    Steuert die PyGame-Ansicht in einem separaten Thread und synchronisiert
    die Weltobjekte mit der Darstellung.

    Attributes:
        settings_controller: Referenz auf den Controller der Tkinter-Settings.
        world: Instanz der Welt mit allen Objekten (Ameisen, Futter etc.).
        game_view: Instanz des PyGame-Fensters.
        running: Boolean, der steuert, ob die Hauptschleife läuft.
        clock: PyGame-Clock-Objekt zur Steuerung der Framerate.
    """
    def __init__(self, settings_controller, world):
        """
        Initialisiert den PyGameController.

        Args:
            settings_controller: Controller für die Tkinter-Settings (UI).
            world: Die Welt, die visualisiert und aktualisiert wird.
        """
        self.name = "PyGameController"
        self.settings_controller = settings_controller  # Hauptcontroller Tk Settings
        self.world = world                              # Enthält alle Weltobjekte (Matrix)
        self.game_view = None                           # Hier Läuft die Matrix im PyGameFenster
        self.running = False                            # Beenden von PyGameFenster durch "False"
        self.clock = pygame.time.Clock()                # Objekt zum Zeit verzögern

    def start_daemon(self):             # Pygame in Neben-Thread wird hier gestartet
        """
        Startet die PyGame-Hauptansicht in einem neuen Daemon-Thread,
        um die GUI nebenläufig laufen zu lassen.
        """
        pygame_thread = threading.Thread(target=self.start_game_view, daemon=True) # args=(),
        pygame_thread.start()                   # Starten im Neben-Thread

    def start_game_view(self):
        """
        Initialisiert PyGame, erstellt das Spiel-Fenster und startet
        die Hauptschleife. Nach Beendigung wird PyGame ordentlich geschlossen.
        """
        pygame.init()      # Initialisiert alle benötigten Pygame-Module (Fenster, Grafik, Eingabe ...).
        self.game_view = view.PyGameWindow()    # Hier Läuft die Matrix im PyGameFenster
        self.loop()                             # Hauptschleife
        pygame.quit()                           # Beendet alle Pygame-Module ordentlich.

    def loop(self):
        """
        Die Hauptschleife, die Events verarbeitet, die Welt aktualisiert
        und die Darstellung rendert, solange `self.running` True ist.
        """
        while self.running:
            self.handle_events()
            if not self.world.world_pause:
                self.update()
                self.game_view.render(self.world)
            self.clock.tick(self.world.clock_tick)  # Nur 10 Bilder pro Sekunde

    def handle_events(self):
        """
        Verarbeitet eingehende PyGame-Events, wie z.B. das Schließen
        des Fensters und setzt `self.running` entsprechend.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        """
        Aktualisiert alle Ameisen in der Welt, überprüft Kollisionen
        mit Futter, aktualisiert die Geruchswelt und synchronisiert
        die UI mit der aktuellen Ameisenanzahl.
        """
        for ant in self.world.ants:
            # if ant.orka <= 0:                                               # Nach bedarf: Ameisen können sterben
            #     # del self.world.ants[self.world.ants.index(ant)]           # Entferne Ameise
            # else:
            ant.move()                                                      # Bewege Ameise
            for food in self.world.foods:                                   # Gehe Foods durch
                if food.get_position() == ant.get_position():               # Ameise hat Futter gefunden
                    # ant.orka += food.calories         # Nach bedarf: Ameisen erhalten Energie vom Futter
                    ant.food_found += 1                                     # Futter gefunden Zähler
                    if self.world.foods.set_food >= len(self.world.foods):  # Wenn Futter gebraucht wird
                        food.set_new_position()                             # Setze Food an neue Position
                    else:
                        del self.world.foods[self.world.foods.index(food)]  # Oder entferne es
                    self.world.update_odor_world()                          # Aktualisiere Geruchsmatrix
        self.update_ant()                                                   # Updates Ant bezogen
        self.update_food()                                                  # Updates Food bezogen

    def update_ant(self):
        """
        Aktualisiert die Anzeige der Ameisenanzahl in der Tkinter-Settings-UI.
        """
        self.settings_controller.tk_view.update_ants_label(len(self.world.ants)) # Setze die Ameisenanzahl im Tk Settings

    def update_food(self):
        """
        Platzhalter-Methode für zukünftige Updates bzgl. Futter,
        derzeit ohne Implementierung.
        """
        pass

class TkSettingsController:                                             # HauptController (Tk Settings)
    """
    Hauptcontroller für die Tkinter-Settings-Oberfläche.

    Verwaltet die Steuerung der Welt, der Ameisen- und Futterobjekte,
    sowie die Verbindung zwischen GUI-Elementen und der Spiellogik.

    Attributes:
        world: Instanz der Welt mit allen Objekten (Ameisen, Futter).
        tk_view: Die Tkinter-basierte Benutzeroberfläche für Settings.
        py_game_controller: Steuerung der PyGame-Ansicht.
        machine_learning_methods: Liste der verfügbaren ML-Methoden.
        ant_strategy: Aktuelle Strategie der Ameisen (random, odor, brain).
        ant_machine_learning: Aktuelle ML-Methode für Ameisen.
    """
    def __init__(self, world, tk_view):
        """
        Initialisiert den Settings-Controller mit Welt und UI.

        Args:
            world: Die Welt mit Ameisen und Futter.
            tk_view: Die Tkinter-Settings-Oberfläche.
        """
        self.name = "SettingsController"
        self.world = world                                              # Enthält alle Weltobjekte (Matrix)
        self.tk_view = tk_view                                          # Tk Settings
        self.py_game_controller = PyGameController(self, world)
        self.machine_learning_methods = ["Monte-Carlo", "Q-Learning"]
        self.ant_strategy = None  # Unterschiedliche Food suche Strategien
        self.ant_machine_learning = None  # Bestimte brain Methode
        self.brain_csv_file = None
        self.ant_ant_selected = "001"

        # Setze Callbacks für UI-Elemente auf Methoden
        self.tk_view.set_show_odor_callback(self.starte_im_thread)
        self.tk_view.set_ant_add_callback(self.set_btn_ant_add)
        self.tk_view.set_food_callback(self.set_btn_food)
        self.tk_view.set_btn_set_speed_callback(self.btn_set_speed)
        self.tk_view.set_btn_pause_callback(self.btn_pause)
        self.tk_view.set_btn_start_callback(self.btn_start)
        self.tk_view.set_btn_random_callback(self.btn_random)
        self.tk_view.set_btn_odor_callback(self.btn_odor)
        self.tk_view.set_show_csv_btn_callback(self.show_csv_btn)
        self.tk_view.set_btn_show_log_callback(self.btn_show_log)
        self.tk_view.set_btn_reset(self.btn_reset)
        self.tk_view.set_btn_brain_callback(self.btn_brain)
        self.tk_view.set_btn_ant_machine_learning(self.btn_ant_machine_learning)
        self.tk_view.set_btn_selected_ant_set(self.btn_selected_ant_set)
        self.tk_view.set_btn_save_ants_callback(self.btn_save_ants)

        # Initialisiere Standardwerte und UI-Status
        self.world.clock_tick = 1                                   # Simulationsgeschwindigkeit
        self.world.world_pause = False
        self.ant_strategy = "brain"             # Unterschiedliche Food suche Strategien ["random" ,"odor", "brain"]
        self.world.ant_machine_learning = "Q-Learning"  # Bestimte brain Methode ["Monte-Carlo", "Q-Learning"]
        # self.world.foods.set_food = 1                               # Anzahl der Essens Objekte setzen
        # self.world.foods.generate_food(self.world.foods.set_food)   # Generiere essen
        # self.world.ants.generate_ant(1, ant_strategy="brain", ant_machine_learning="Q-Learning")      # Einen mit Brain
        self.tk_view.update_log_widget_text("Hier Werden die Internen Daten, der Ameise angezeigt.")


        # Konfiguration der Tk Elemente
        self.tk_view.update_ent_set_food(5) # Setze Food Entry
        self.tk_view.update_ent_ant_add(1)  # Setze Ant Entry

        self.tk_view.update_cmb_ant_machine_learning(self.machine_learning_methods, self.world.ant_machine_learning)
        self.tk_view.update_cmb_selected_ant(["001"],"001")

        self.update_settings_window()       # Aktualisiere Labels
        self.world.update_odor_world()      # Setze Geruch

    def starte_im_thread(self):
        """
        Startet die Darstellung des Geruchsfelds in einem separaten Thread,
        um die UI nicht zu blockieren.
        """
        threading.Thread(target=self.stelle_py_plot_array_dar, daemon=True).start()

    def stelle_py_plot_array_dar(self):
        """
        Pausiert die Simulation und zeigt das Geruchsfeld mittels Matplotlib an.
        """
        self.world.world_pause = True                                              # Pausiere die Matrix
        self.tk_view.after(0, lambda: view.FoodOdorPyPlot(self.world.world_array)) # Stelle PyPlot dar

    def set_btn_ant_add(self):
        """
        Fügt zusätzliche Ameisen entsprechend des Eingabefeldes hinzu,
        aktualisiert die Anzeige und gibt Debug-Infos aus, setzt Callback.
        """
        self.world.ants.generate_ants(int(self.tk_view.get_ent_ant_add_value()), self.ant_strategy, self.ant_machine_learning)
        if len(self.world.ants.show_ants()) > 0:            # Wenn Ameisen vorhanden sind
            # self.btn_selected_ant_set()                     # Speicher den ausgewählten Wert
            self.tk_view.update_cmb_selected_ant([ant.name for ant in self.world.ants], self.ant_ant_selected) # Aktualisiere Combobox
            # self.tk_view.update_log_widget_text(self.world.ants.show_ants()[int(self.ant_ant_selected) - 1].log_collector.get_formatted_info())
        self.tk_view.update_ent_ant_add(0)                              # Setze Ant Entry auf Null
        self.update_settings_window()
        for ant in self.world.ants.show_ants():                         # Hier wird der Callback für Textfeld gesetzt
            ant.log_collector.update_log_collector_callback(self.update_log_collector_text)

    def set_btn_food(self):
        """
        Setzt die gewünschte Anzahl Futterobjekte, generiert bei Bedarf neues Futter,
        aktualisiert Geruchsmatrix und die Anzeige.
        """
        if int(self.tk_view.get_ent_set_food_value()) > int(len(self.world.foods)): # Wenn die Zahl im entry gröser ist wie Foodobjekte
            self.world.foods.generate_food(int(self.tk_view.get_ent_set_food_value()) - int(len(self.world.foods))) # Erstelle mehr Food
        self.world.foods.set_food = int(self.tk_view.get_ent_set_food_value())      # Setze geforderte Essen Variable auf entry
        self.world.update_odor_world()                                              # Aktualisiere Geruchsmatrix
        self.update_settings_window()                                               # Aktualisiere Window
        # self.tk_view.update_ent_set_food(self.world.foods.set_food)  # Setze Food entry

    def btn_set_speed(self):
        """
        Setzt die Simulationsgeschwindigkeit anhand des Sliders.
        """
        self.world.clock_tick = self.tk_view.get_sld_speed_value()    # Setze Geschwindigkeit von dem slider

    def btn_pause(self):
        """
        Schaltet die Pause um (an/aus).
        """
        if self.world.world_pause:
            self.world.world_pause = False  # Hebe Pause auf
        else:
            self.world.world_pause = True   # Setze Pause

    def btn_start(self):
        """
        Startet die Simulation, fügt Ameisen und Futter gemäß Vorgabe hinzu
        und startet den PyGame-Thread, falls noch nicht gestartet.
        """
        if self.ant_strategy == "brain":
            self.btn_ant_machine_learning()         # Setze Machine Learning Methode
        self.set_btn_food()                     # Setze das voreingestellte Futter
        self.set_btn_ant_add()                  # Füge die voreingestellten Ameisen ein
        self.world.world_pause = False          # Hebe Pause auf
        if not self.py_game_controller.running:         # Wenn schon läuft, nicht starten
            self.py_game_controller.running = True
            self.py_game_controller.start_daemon()      # Starte PyGame als Daemon

    def btn_random(self):
        """
        Setzt die Ameisenstrategie auf "random" und aktualisiert das UI.
        """
        self.ant_strategy = "random"
        self.update_settings_window()

    def btn_odor(self):
        """
        Setzt die Ameisenstrategie auf "odor" (Geruch folgen) und aktualisiert das UI.
        """
        self.ant_strategy = "odor"
        self.update_settings_window()

    def show_csv_btn(self):
        """
        Öffnet ein neues Fenster zur Anzeige der gespeicherten Q-Learning CSV-Datei,
        falls die Strategie 'brain' aktiv ist.
        """
        if self.ant_strategy == "brain":
            view.CSVViewer(csv_path=self.brain_csv_file).mainloop()
        else:
            print("In diesem Modus nicht Möglich")

    def btn_show_log(self):
        """
        Öffnet ein neues Fenster zur Anzeige der Ereignissen die stattgefunden haben.
        """
        self.world.world_pause = True
        inhalt = self.world.ants.show_ants()[int(self.ant_ant_selected)-1].log_collector.get_all_periods()
        view.TextEditor(inhalt).mainloop()

    def btn_reset(self):
        """
        Setzt alles zurück, ermöglicht neustart ohne PyGame beenden zu müssen.
        """
        self.world.world_pause = True           # Setze Pause
        self.world.ants.clear()                 # Lösche alle Ameisen
        self.tk_view.update_ent_set_food(0)     # Setze Food entry auf Null
        self.world.foods.clear()                # Lösche alle Foods
        self.set_btn_food()                     # Aktualisiere Foods
        self.tk_view.update_cmb_selected_ant(["001"],"001")
        self.world.world_pause = False          # Starte wieder

    def btn_brain(self):
        """
        Setzt die Ameisenstrategie auf "brain" (ML-basiert) und aktualisiert das UI.
        """
        self.ant_strategy = "brain"
        self.ant_machine_learning = "Monte-Carlo"
        self.update_settings_window()

    def btn_ant_machine_learning(self):
        """
        Ändert die eingesetzte Machine Learning Methode für die Ameisen
        anhand der ComboBox-Auswahl und aktualisiert die ComboBox.
        """
        self.ant_strategy = "brain"
        self.ant_machine_learning = self.tk_view.get_cmb_ant_machine_learning_value() # Setze Machine Learning Methode
        self.update_settings_window()

    def btn_selected_ant_set(self):
        """
        Ändert die ausgewählte Ameise für die Textanzeige
        anhand der ComboBox-Auswahl und aktualisiert die ComboBox.
        """
        self.ant_ant_selected = self.tk_view.get_cmb_selected_ant_value()
        self.update_log_collector_text()

    def btn_save_ants(self):
        """
        Speichert den Lernerfolg der Ameisen (Q-Werte) in eine CSV-Datei,
        wenn die Bedingungen erfüllt sind (einzelne Ameise mit Brain).
        """
        self.world.world_pause = True
        if self.world.ants.show_ants()[0].name == "brain" and len(self.world.ants.show_ants()) == 1: # Nur in diesem Modus möglich
            q_items = self.world.ants.show_ants()[0].brain.out_brain()
            model.DataStorage.save_q_to_csv(q_items, self.brain_csv_file)
        else:
            print("Nur mit Brain und einer Ameise Möglich")

    def update_settings_window(self):
        """
        Aktualisiert die Grundlegende Labels im tk_view
        """
        self.tk_view.sld_speed.set(self.world.clock_tick)           # Setze Geschwindigkeits Slider
        self.tk_view.update_ants_label(len(self.world.ants))        # Setze Ameisenanzahl Label
        self.tk_view.update_food_label(self.world.foods.set_food)   # Setze Futteranzahl Label
        self.tk_view.update_lbl_set_brain(self.ant_strategy)        # Setze label für Strategie

        if self.ant_strategy == "brain":
            self.tk_view.update_lbl_set_machine_learning(self.ant_machine_learning) # Setze label für Machine Learning
            if self.ant_machine_learning == "Monte-Carlo":
                self.brain_csv_file = config.MONTE_CARLO_FILE
            elif self.ant_machine_learning == "Q-Learning":
                self.brain_csv_file = config.Q_LEARNING_FILE
        else:
            self.tk_view.update_lbl_set_machine_learning(".")

    def update_log_collector_text(self):
        """
        Aktualisiert den Text des Text Widget im tk_view
        Funktioniert über Callback
        """
        if self.ant_ant_selected == "None": # Wenn keine Ameisen vorhanden sind
            text = "!!!Keine Ameise ausgewählt!!!"
            self.tk_view.update_log_widget_text(text)
        else:
            text = self.world.ants.show_ants()[int(self.ant_ant_selected) - 1].log_collector.get_formatted_info()
            self.tk_view.update_log_widget_text(text)



# === Testpoint ===
if __name__ == "__main__":
    main.main()