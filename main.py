#!/usr/bin/env python3
"""
Simulation einer Ameise

Dieses Modul startet die Hauptanwendung für eine Ameisen-Simulation mit grafischer Benutzeroberfläche.
Die Anwendung folgt dem MVC-Prinzip (Model-View-Controller) und dient als Lernprojekt für künstliche Intelligenz.

Autor:       Artur Lamparter <arturlamparter@web.de>
Projektziel: Demonstration KI-gesteuerter Agenten im simulierten Umfeld
"""
__author__ = "Artur Lamparter <arturlamparter@web.de>"
__version__ = "0.2.1" #git tag v0.2.1

# --- Externe Modulintegration ---
import sys

import view         # Enthält die GUI-Elemente (Tkinter-Views)
import controller   # Steuert die Verbindung zwischen Model und View (Eventhandling etc.)
import model        # Repräsentiert die Datenstruktur der Welt (z.B. die Matrix der Ameise)


def main() -> None:
    """
    Startet die Hauptanwendung.

    - Erstellt ein Modell-Objekt (Welt/Matrix)
    - Initialisiert das GUI-Fenster
    - Verknüpft Controller mit Model und View
    - Startet die Tkinter Hauptschleife
    """

    # Initialisiere Weltobjekt mit Matrixdaten und Simulationslogik
    world_obj = model.World()

    # Erzeuge das GUI-Hauptfenster mit Steuerelementen
    tk_settings_view_obj = view.TkSettingsWindow()

    # Koppele das Modell und die View über einen Controller (Ist nur für Tk zuständig)
    controller.TkSettingsController(world_obj, tk_settings_view_obj)

    # Starte die GUI (muss im Hauptthread laufen)
    tk_settings_view_obj.mainloop()

    sys.exit()  # Beendet vollständig. Sicherer Abschluss wenn du das Programm aus einer IDE startest.


# --- Einstiegspunkt für das Programm ---
if __name__ == "__main__":
    main()  # Starte die App