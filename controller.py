"""
Dieses Modul `controller` enthält alle Steuerungs-Elemente des Projekts.

Es umfasst die Implementierungen für die Steuerung allen Details. Dieses Modul steuert das gesamte Projekt.
Benutzeroberfläche der Einstellungen und die Darstellung der Ameisen-Simulation.

Autor: Artur Lamparter <arturlamparter@web.de>
"""
import threading

import pygame

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
        self.game_view = view.PyGameWindow(self.world)    # Hier Läuft die Matrix im PyGameFenster
        self.loop()                             # Hauptschleife
        pygame.quit()                           # Beendet alle Pygame-Module ordentlich.

    def loop(self):
        """
        Die Hauptschleife, die Events verarbeitet, die Welt aktualisiert
        und die Darstellung rendert, solange `self.running` True ist.
        """
        while self.running:
            self.handle_events()
            if not self.world.world_pause:              # Pause
                self.update()
                self.game_view.render()

            if self.world.step:                         # Schritt
                self.world.step = False
                self.world.world_pause = True

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
            if not ant.out_of_action: ant.move()                                                      # Bewege Ameise
            for food in self.world.foods:                                   # Gehe Foods durch
                if food.get_position() == ant.get_position():               # Ameise hat Futter gefunden
                    ant.orka += food.calories         # Nach bedarf: Ameisen erhalten Energie vom Futter
                    ant.food_found += 1                                     # Futter gefunden Zähler
                    if self.world.foods.set_food >= len(self.world.foods):  # Wenn Futter gebraucht wird
                        food.set_new_position()                             # Setze Food an neue Position
                    else:
                        del self.world.foods[self.world.foods.index(food)]  # Oder entferne es
            if ant.orka <= 0:
                ant.out_of_action = True
            else:
                ant.orka -= 1
            self.world.update_odor_world()                          # Aktualisiere Geruchsmatrix
        self.update_ant()                                                   # Updates Ant bezogen

    def update_ant(self):
        """
        Aktualisiert die Anzeige der Ameisenanzahl in der Tkinter-Settings-UI.
        """
        self.settings_controller.tk_settings_window.update_ants_label(len(self.world.ants)) # Setze die Ameisenanzahl im Tk Settings

class TkSettingsController:                                             # HauptController (Tk Settings)
    """
    Hauptcontroller für die Tkinter-Settings-Oberfläche.

    Verwaltet die Steuerung der Welt, der Ameisen- und Futterobjekte,
    sowie die Verbindung zwischen GUI-Elementen und der Spiellogik.

    Attributes:
        world: Instanz der Welt mit allen Objekten (Ameisen, Futter).
        tk_settings_window: Die Tkinter-basierte Benutzeroberfläche für Settings.
        py_game_controller: Steuerung der PyGame-Ansicht.
        machine_learning_methods: Liste der verfügbaren ML-Methoden.
        ant_strategy: Aktuelle Strategie der Ameisen (random, odor, brain).
        ant_machine_learning: Aktuelle ML-Methode für Ameisen.
    """
    def __init__(self, settings_window):
        """
        Initialisiert den Settings-Controller mit Welt und UI.

        Args:
            world: Die Welt mit Ameisen und Futter.
            settings_window: Die Tkinter-Settings-Oberfläche.
        """
        self.name = "SettingsController"
        self.world = model.World()                                      # Enthält alle Weltobjekte (Matrix)
        self.tk_settings_window = settings_window                                          # Tk Settings
        self.py_game_controller = PyGameController(self, self.world)
        self.machine_learning_methods = ["Monte-Carlo", "Q-Learning", "Perzeptron"]
        self.ant_strategy = model.ANT_STRATEGY  # Unterschiedliche Food suche Strategien ["random" ,"odor", "brain"]
        self.ant_machine_learning = model.ANT_MACHINE_LEARNING  # Bestimte brain Methode
        self.cmb_ant_selected = "000"

        # Setze Callbacks für UI-Elemente auf Methoden
        self.tk_settings_window.set_show_odor_callback(self.starte_im_thread)
        self.tk_settings_window.set_btn_brain_callback(self.btn_brain)
        self.tk_settings_window.set_btn_step_cb(self.btn_step_click)
        self.tk_settings_window.set_btn_ant_settings_cb(self.btn_ant_settings)
        self.tk_settings_window.set_ant_add_callback(self.set_btn_ant_add)
        self.tk_settings_window.set_btn_food_settings_cb(self.btn_food_settings)
        self.tk_settings_window.set_food_callback(self.set_btn_food)
        self.tk_settings_window.set_btn_set_speed_callback(self.btn_set_speed)
        self.tk_settings_window.set_btn_pause_callback(self.btn_pause)
        self.tk_settings_window.set_btn_start_callback(self.btn_start)
        self.tk_settings_window.set_btn_random_callback(self.btn_random)
        self.tk_settings_window.set_btn_odor_callback(self.btn_odor)
        self.tk_settings_window.set_show_csv_btn_callback(self.show_csv_btn)
        self.tk_settings_window.set_btn_show_log_callback(self.btn_show_log)
        self.tk_settings_window.set_btn_reset(self.btn_reset)
        self.tk_settings_window.set_btn_training_cb(self.btn_training)
        self.tk_settings_window.set_btn_save_ants_callback(self.btn_save_ants)

        # Initialisiere Standardwerte und UI-Status
        self.world.clock_tick = 60                                   # Simulationsgeschwindigkeit
        self.world.world_pause = False
        self.tk_settings_window.update_sld_speed(self.world.clock_tick)         # Setze Geschwindigkeits Slider

        # self.world.ant_machine_learning = "Q-Learning"  # Bestimte brain Methode ["Monte-Carlo", "Q-Learning"]
        # self.world.foods.set_food = 1                               # Anzahl der Essens Objekte setzen
        # self.world.foods.generate_food(self.world.foods.set_food)   # Generiere essen
        # self.world.ants.generate_ant(1, ant_strategy="brain", ant_machine_learning="Q-Learning")      # Einen mit Brain
        self.tk_settings_window.update_log_widget_text("Hier Werden die Internen Daten, der Ameise angezeigt.")


        # Konfiguration der Tk Elemente
        self.tk_settings_window.update_ent_set_food(100) # Setze Food Entry
        self.tk_settings_window.update_ent_ant_add(1)  # Setze Ant Entry

        # --- Alles aktualisieren ---
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
        self.tk_settings_window.after(0, lambda: view.FoodOdorPyPlot(self.world.world_array)) # Stelle PyPlot dar

    def btn_brain(self):
        """
        Setzt die Ameisenstrategie auf "brain" (ML-basiert) und aktualisiert das UI.
        """
        self.ant_strategy = "brain"
        self.ant_machine_learning = "Monte-Carlo"
        self.update_settings_window()

    def btn_step_click(self):
        self.world.world_pause = False
        self.world.step = True

    def btn_ant_settings(self):
        ant_settings_obj = view.AntSettingsWindow(self.tk_settings_window)
        ant_settings_values = {"RANDOM_COLOR": model.RANDOM_COLOR,
                               "ODOR_COLOR": model.ODOR_COLOR,
                               "MONTE_CARLO_COLOR": model.MONTE_CARLO_COLOR,
                               "Q_LEARNING_COLOR": model.Q_LEARNING_COLOR,
                               "ORKA": model.ORKA,
                               "GENERATION": False}
        ant_settings_obj.update_settings(ant_settings_values)
        ant_settings_obj.set_btn_save_cb(lambda: model.DataStorage().save_settings(ant_settings_obj.get_settings()))

    def set_btn_ant_add(self):
        """
        Fügt zusätzliche Ameisen entsprechend des Eingabefeldes hinzu,
        aktualisiert die Anzeige und gibt Debug-Infos aus, setzt Callback.
        """
        self.world.ants.generate_ants(int(self.tk_settings_window.get_ent_ant_add_value()), self.ant_strategy,
                                      self.ant_machine_learning, self.tk_settings_window.get_csv_load())
        if len(self.world.ants.show_ants()) > 0:            # Wenn Ameisen vorhanden sind
            self.cmb_ant_selected = self.world.ants.show_ants()[-1].name
        self.tk_settings_window.update_ent_ant_add(1)                              # Setze Ant Entry auf Eins
        self.update_settings_window()
        for ant in self.world.ants.show_ants():                         # Hier wird der Callback für Textfeld gesetzt
            ant.log_collector.update_log_collector_callback(self.update_log_collector_text)

    def btn_food_settings(self):
        food_settings_obj = view.FoodSettingsWindow(self.tk_settings_window)
        food_settings_values = {"FOOD_RANDOM_COLOR": model.FOOD_RANDOM_COLOR,
                               "FOOD_FIXED_SIZE_COLOR": model.FOOD_FIXED_SIZE_COLOR,
                               "FOOD_RANGE": model.FOOD_RANGE,
                               "RANDOM_FOOD": model.RANDOM_FOOD}
        food_settings_obj.update_settings(food_settings_values)
        food_settings_obj.set_btn_save_cb(lambda: model.DataStorage().save_settings(food_settings_obj.get_settings()))

    def set_btn_food(self):
        """
        Setzt die gewünschte Anzahl Futterobjekte, generiert bei Bedarf neues Futter,
        aktualisiert Geruchsmatrix und die Anzeige.
        """
        if int(self.tk_settings_window.get_ent_set_food_value()) > int(len(self.world.foods)): # Wenn die Zahl im entry gröser ist wie Foodobjekte
            self.world.foods.generate_food(int(self.tk_settings_window.get_ent_set_food_value()) - int(len(self.world.foods))) # Erstelle mehr Food
        self.world.foods.set_food = int(self.tk_settings_window.get_ent_set_food_value())      # Setze geforderte Essen Variable auf entry
        self.world.update_odor_world()                                              # Aktualisiere Geruchsmatrix
        self.update_settings_window()                                               # Aktualisiere Window
        # self.tk_view.update_ent_set_food(self.world.foods.set_food)  # Setze Food entry

    def btn_set_speed(self):
        """
        Setzt die Simulationsgeschwindigkeit anhand des Sliders.
        """
        self.world.clock_tick = self.tk_settings_window.get_sld_speed_value()    # Setze Geschwindigkeit von dem slider

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
            self.cmb_ant_machine_learning()         # Setze Machine Learning Methode
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
        self.ant_machine_learning = "Keine"
        self.update_settings_window()

    def btn_odor(self):
        """
        Setzt die Ameisenstrategie auf "odor" (Geruch folgen) und aktualisiert das UI.
        """
        self.ant_strategy = "odor"
        self.ant_machine_learning = "Keine"
        self.update_settings_window()

    def show_csv_btn(self):
        """
        Öffnet ein neues Fenster zur Anzeige der gespeicherten Q-Learning CSV-Datei,
        falls die Strategie 'brain' aktiv ist.
        """
        self.world.world_pause = True
        ant = self.world.ants.get_ant(self.cmb_ant_selected)
        if ant:
            if ant.brain.ant_strategy == "brain":
                 view.CSVViewer(csv_path=ant.brain.data_file).mainloop()
            else:
                model.logger.info("Nur im Brain Modus Möglich.")
        else:
            model.logger.error("Ameise nicht gefunden.")

    def btn_show_log(self):
        """
        Öffnet ein neues Fenster zur Anzeige der Ereignissen die stattgefunden haben.
        """
        self.world.world_pause = True
        if int(self.cmb_ant_selected) > 0:
            inhalt = self.world.ants.show_ants()[int(self.cmb_ant_selected) - 1].log_collector.get_all_periods()
            view.TextEditor(inhalt).mainloop()
        else:
            print("Keine Daten Vorhanden")

    def btn_reset(self):
        """
        Setzt alles zurück, ermöglicht neustart ohne PyGame beenden zu müssen.
        """
        self.world.world_pause = True           # Setze Pause
        self.world.ants.clear()                 # Lösche alle Ameisen
        self.tk_settings_window.update_ent_set_food(0)     # Setze Food entry auf Null
        self.world.foods.clear()                # Lösche alle Foods
        self.set_btn_food()                     # Aktualisiere Foods
        self.world.world_pause = False          # Starte wieder
        self.update_settings_window()

    def btn_training(self):
        brain_trainings_obj = view.BrainTrainingsWindow(self.tk_settings_window)
        # brain_trainings_values = {"FOOD_RANDOM_COLOR": model.FOOD_RANDOM_COLOR,
        #                         "FOOD_FIXED_SIZE_COLOR": model.FOOD_FIXED_SIZE_COLOR,
        #                         "FOOD_RANGE": model.FOOD_RANGE,
        #                         "RANDOM_FOOD": model.RANDOM_FOOD}
        # food_settings_obj.update_settings(brain_trainings_values)
        brain = model.Brain(name="001", ant_strategy="brain", ant_machine_learning="Perzeptron", csv_load=False)
        brain_trainings_obj.set_btn_go_cb(lambda: self.test(brain_trainings_obj.get_settings(), brain))


    def test(self, settings, brain):
        brain.train_perzeptron(settings)
        self.tk_settings_window.update_log_widget_text(brain.log_collector.get_formatted_info())

    def cmb_ant_machine_learning(self):
        """
        Ändert die eingesetzte Machine Learning Methode für die Ameisen
        anhand der ComboBox-Auswahl und aktualisiert die ComboBox.
        """
        self.ant_strategy = "brain"
        self.ant_machine_learning = self.tk_settings_window.get_cmb_ant_machine_learning_value() # Setze Machine Learning Methode
        self.update_settings_window()

    def cmb_selected_ant_set(self):
        """
        Ändert die ausgewählte Ameise für die Textanzeige
        anhand der ComboBox-Auswahl und aktualisiert die ComboBox.
        """
        self.cmb_ant_selected = self.tk_settings_window.get_cmb_selected_ant_value()
        self.update_log_collector_text()

    def btn_save_ants(self):
        """
        Speichert den Lernerfolg der Ameisen (Q-Werte) in eine CSV-Datei,
        wenn die Bedingungen erfüllt sind (einzelne Ameise mit Brain).
        """
        self.world.world_pause = True
        ant = self.world.ants.get_ant(self.cmb_ant_selected)
        if ant: ant.brain.save_brain_data()

    def update_settings_window(self):
        """
        Aktualisiert die Grundlegende Labels im tk_view
        """
        self.world.clock_tick = self.tk_settings_window.get_sld_speed_value()  # Setze Geschwindigkeit von dem slider
        self.tk_settings_window.update_ants_label(len(self.world.ants))        # Setze Ameisenanzahl Label
        self.tk_settings_window.update_food_label(self.world.foods.set_food)   # Setze Futteranzahl Label
        self.tk_settings_window.update_lbl_set_brain(self.ant_strategy)        # Setze label für Strategie
        self.tk_settings_window.update_cmb_selected_ant([ant.name for ant in self.world.ants], self.cmb_ant_selected, self.cmb_selected_ant_set)
        if self.ant_strategy == "brain":
            self.tk_settings_window.update_lbl_set_machine_learning(self.ant_machine_learning) # Setze label für Machine Learning
            self.tk_settings_window.update_cmb_ant_machine_learning(self.machine_learning_methods, self.ant_machine_learning, self.cmb_ant_machine_learning)
        else:
            self.tk_settings_window.update_lbl_set_machine_learning(".")

    def update_log_collector_text(self):
        """
        Aktualisiert den Text des Text Widget im tk_view
        Funktioniert über Callback
        """
        if self.cmb_ant_selected == "None": # Wenn keine Ameisen vorhanden sind
            text = "!!!Keine Ameise ausgewählt!!!"
            self.tk_settings_window.update_log_widget_text(text)
        else:
            text = self.world.ants.show_ants()[int(self.cmb_ant_selected) - 1].log_collector.get_formatted_info()
            self.tk_settings_window.update_log_widget_text(text)

# === Testpoint ===
if __name__ == "__main__":
    main.main()