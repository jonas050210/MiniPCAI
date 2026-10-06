"""Generate the versioned MiniPCAI intent dataset.

The dataset is the single source of truth for training the intent classifier.
It is a versioned JSONL file (``data/intent_dataset.v1.jsonl``) where each row
has the following fields:

* ``id``        - stable unique identifier (``<label>-<counter>``)
* ``text``      - the German user utterance
* ``label``     - one of the 12 action intents or ``unknown``
* ``category``  - ``normal`` | ``typo`` | ``unknown`` | ``ambiguous``
* ``version``   - dataset version (must match the file name)

Category semantics:

* ``normal``    - a clear, correctly spelled in-domain request.
* ``typo``      - the same kind of request with realistic typing mistakes.
* ``unknown``   - out-of-domain or unsupported requests (label ``unknown``).
* ``ambiguous`` - requests that could map to several intents or targets and
  must trigger a clarification instead of a guess (label ``unknown``).

Ambiguous examples are trained towards ``unknown`` on purpose: the safe answer
for an unclear request is to ask for clarification, never to act.  Runtime
ambiguity is additionally detected through the confidence/margin gates of the
pipeline.

The generator is deterministic (fixed random seed) so the committed dataset can
always be reproduced:

    python scripts/generate_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATASET_VERSION = 1

# ---------------------------------------------------------------------------
# Handwritten German example utterances per action intent (category "normal")
# ---------------------------------------------------------------------------

NORMAL_EXAMPLES: dict[str, list[str]] = {
    "open_app": [
        "öffne notepad",
        "starte notepad",
        "notepad öffnen",
        "mach notepad auf",
        "öffne bitte notepad",
        "kannst du notepad starten",
        "ich möchte notepad öffnen",
        "starte den editor",
        "öffne den editor",
        "editor starten bitte",
        "mach den texteditor auf",
        "starte das malprogramm",
        "öffne paint",
        "paint öffnen",
        "öffne mal paint",
        "starte den taschenrechner",
        "öffne den taschenrechner",
        "taschenrechner öffnen",
        "mach den taschenrechner auf",
        "bitte den taschenrechner starten",
        "starte firefox",
        "öffne firefox",
        "starte den browser",
        "öffne den browser",
        "browser öffnen",
        "mach den browser auf",
        "starte chrome",
        "öffne chrome bitte",
        "öffne edge",
        "starte word",
        "öffne word",
        "mach word auf",
        "starte excel",
        "öffne excel bitte",
        "excel starten",
        "öffne powerpoint",
        "starte outlook",
        "öffne das mailprogramm",
        "starte den medienplayer",
        "öffne vlc",
        "starte notepad bitte",
        "notepad starten",
        "öffne mir mal notepad",
        "könntest du den editor öffnen",
        "bitte editor öffnen",
        "öffne schnell den editor",
        "starte das programm notepad",
        "programm editor öffnen",
        "öffne die systemsteuerung",
        "starte das zeichenprogramm",
        "würdest du notepad starten",
        "ich brauche den editor",
        "öffne den rechner mal",
        "starte chrome bitte",
        "chrome starten",
        "öffne chrome mal",
        "mach chrome auf",
        "chrome öffnen bitte",
        "starte bitte chrome",
        "starte word bitte",
        "word öffnen",
        "mach word auf bitte",
        "starte excel bitte",
        "excel öffnen",
        "öffne mal excel",
        "starte powerpoint bitte",
        "powerpoint öffnen",
        "öffne teams",
        "starte teams",
        "öffne vlc bitte",
        "vlc starten",
        "starte gimp",
        "öffne das bildbearbeitungsprogramm",
        "starte mal den browser",
        "browser mal aufmachen",
        "notepad soll starten",
        "öffne firefox und chrome",
        "starte editor und browser",
        "videoanruf starten",
        "starte eine videokonferenz mit kollegen",
        "starte es",
    ],
    "close_app": [
        "schließe notepad",
        "beende notepad",
        "notepad schließen",
        "mach notepad zu",
        "notepad beenden",
        "beende bitte den editor",
        "schließ den editor",
        "editor zumachen",
        "beende firefox",
        "firefox schließen",
        "mach den browser zu",
        "beende den browser",
        "browser schließen bitte",
        "stoppe den browser",
        "beende word",
        "word schließen",
        "mach excel zu",
        "schließe excel bitte",
        "beende das programm notepad",
        "beende die app",
        "notepad soll sich schließen",
        "kannst du notepad schließen",
        "bitte notepad beenden",
        "mach das textprogramm zu",
        "beende paint",
        "paint schließen",
        "schließe den taschenrechner",
        "beende den taschenrechner bitte",
        "beenden sie notepad",
        "könntest du firefox beenden",
        "schließ bitte word",
        "outlook schließen",
        "beende outlook",
        "mach den medienplayer zu",
        "editor beenden",
        "das programm soll geschlossen werden",
        "schließe das fenster vom editor",
        "beende mal bitte den browser",
        "excel soll zugehen",
        "stoppe das programm",
        "stoppe notepad",
        "stoppe den editor",
        "stoppe word bitte",
        "stoppe excel",
        "stopp den browser",
        "stopp notepad",
        "mach vlc zu",
        "mach den player zu",
        "mach paint zu",
        "mach word zu",
        "excel soll beendet werden",
        "notepad soll geschlossen werden",
        "word soll sich schließen",
        "der browser soll zugehen",
        "schließ firefox",
        "schließ chrome bitte",
        "schließ mal den editor",
        "mach zu",
        "schließ das",
        "beende das",
        "mach das andere zu",
        "beende die sache",
    ],
    "open_url": [
        "öffne wikipedia",
        "öffne die webseite wikipedia",
        "geh auf wikipedia",
        "öffne die seite wikipedia",
        "wikipedia öffnen",
        "mach wikipedia auf",
        "starte die webseite wikipedia",
        "öffne wikipedia im browser",
        "besuche wikipedia",
        "öffne google",
        "geh auf google",
        "google öffnen",
        "öffne die suchmaschine",
        "starte google",
        "öffne die website google",
        "wechsle zu wikipedia",
        "zeig mir die wikipedia webseite",
        "öffne bitte wikipedia",
        "kannst du wikipedia öffnen",
        "ruf wikipedia auf",
        "ruf die seite wikipedia auf",
        "öffne youtube",
        "geh auf youtube",
        "youtube öffnen",
        "öffne die github webseite",
        "github öffnen",
        "mach github auf",
        "wikipedia im browser öffnen",
        "öffne mal die suchmaschine",
        "könntest du wikipedia öffnen",
        "ich möchte auf wikipedia gehen",
        "zeig mir google",
        "starte bitte die webseite",
        "öffne die nachschlagewerk seite",
        "öffne die webseite",
        "öffne die webseite github bitte",
        "github im browser öffnen",
        "ruf google auf",
        "ruf youtube auf bitte",
        "öffne bitte youtube",
        "mach die suchmaschine auf",
        "zeig mir youtube",
    ],
    "open_file": [
        "öffne die notizen",
        "öffne die datei notizen",
        "mach die notizen auf",
        "öffne mein protokoll",
        "die datei notizen öffnen",
        "öffne bitte die notizen",
        "öffne das protokoll",
        "zeig mir die notizen",
        "öffne die rechnung",
        "öffne die datei rechnung",
        "mach die datei protokoll auf",
        "kannst du die notizen öffnen",
        "öffne mir die notizen",
        "die rechnung öffnen bitte",
        "öffne das dokument",
        "öffne die datei bericht",
        "bericht öffnen",
        "mach den bericht auf",
        "öffne meine notizen datei",
        "notizen datei öffnen",
        "öffne die pdf datei",
        "öffne die tabelle",
        "tabelle öffnen",
        "öffne die präsentation",
        "mach die präsentation auf",
        "bitte die datei protokoll öffnen",
        "die datei rechnung aufmachen",
        "könntest du das protokoll öffnen",
        "zeig mir den bericht",
        "öffne mal die notizen",
        "ich möchte die notizen sehen",
        "mach mir die rechnung auf",
        "öffne das textdokument",
        "die tabelle aufmachen",
        "öffne meine rechnung",
        "öffne die notizen im editor",
        "zeig mir die tabelle",
        "öffne die datei tabelle bitte",
        "mach mir die präsentation auf",
    ],
    "open_folder": [
        "öffne die downloads",
        "öffne den download ordner",
        "öffne den ordner downloads",
        "downloads öffnen",
        "mach den ordner downloads auf",
        "öffne dokumente",
        "öffne den ordner dokumente",
        "dokumente öffnen",
        "mach die downloads auf",
        "öffne meine bilder",
        "öffne den bilder ordner",
        "bilder öffnen",
        "öffne den ordner bilder bitte",
        "zeig mir den downloads ordner",
        "öffne den ordner",
        "öffne die musik",
        "musik ordner öffnen",
        "öffne den videos ordner",
        "videos öffnen",
        "mach den dokumente ordner auf",
        "öffne bitte den ordner downloads",
        "kannst du den ordner dokumente öffnen",
        "öffne mir die downloads",
        "den ordner bilder öffnen",
        "dokumente aufmachen",
        "öffne den download-ordner",
        "zeig mir die downloads",
        "mach den ordner auf",
        "öffne den ordner musik",
        "ich möchte in die downloads",
        "öffne mal die dokumente",
        "der ordner bilder soll aufgehen",
        "download ordner aufmachen",
        "öffne die bilder bitte",
    ],
    "find_file": [
        "finde die datei rechnung",
        "suche die datei rechnung",
        "suche rechnung",
        "finde rechnung",
        "wo ist die datei rechnung",
        "suche nach dem protokoll",
        "finde das protokoll",
        "suche das protokoll",
        "wo ist mein protokoll",
        "finde die notizen",
        "suche die notizen",
        "wo sind meine notizen",
        "suche die datei notizen",
        "suche nach rechnung",
        "suche nach der rechnung",
        "finde mir die rechnung",
        "kannst du die datei rechnung finden",
        "suche bitte das protokoll",
        "durchsuche die dateien nach rechnung",
        "suche bericht",
        "finde bericht",
        "wo liegt das protokoll",
        "wo liegt die datei bericht",
        "finde die datei bericht",
        "suche die tabelle",
        "finde die präsentation",
        "wo ist die präsentation",
        "finde die datei mit dem bericht",
        "suche nach der tabelle",
        "wo habe ich die rechnung gespeichert",
        "finde die pdf rechnung",
        "suche die pdf datei rechnung",
        "such mal nach dem protokoll",
        "finde bitte die notizen datei",
        "könntest du die rechnung finden",
        "suche mir die präsentation",
    ],
    "sys_cpu": [
        "wie hoch ist die cpu auslastung",
        "cpu auslastung anzeigen",
        "zeig mir die prozessorauslastung",
        "cpu auslastung",
        "wie stark ist die cpu gerade ausgelastet",
        "cpu last",
        "prozessor auslastung zeigen",
        "wie busy ist die cpu",
        "wie viel prozent nutzt die cpu",
        "cpu nutzung anzeigen",
        "zeig die cpu last",
        "aktuelle prozessorauslastung",
        "wie ist die auslastung vom prozessor",
        "prozessorlast anzeigen",
        "wie ausgelastet ist der prozessor",
        "sag mir die cpu auslastung",
        "cpu überprüfen",
        "prüfe die prozessor auslastung",
        "wie läuft die cpu",
        "prozessor status",
        "aktuelle cpu werte",
        "cpu kern auslastung zeigen",
        "wie viel macht die cpu gerade",
        "zeige die aktuelle prozessorlast",
        "cpu auslastung bitte",
        "wie stark ist die prozessorlast",
    ],
    "sys_ram": [
        "wie viel arbeitsspeicher ist belegt",
        "arbeitsspeicher auslastung anzeigen",
        "ram auslastung",
        "wie viel ram ist frei",
        "zeig mir die ram auslastung",
        "arbeitsspeicher anzeigen",
        "wie viel arbeitsspeicher ist noch frei",
        "ram status",
        "wie voll ist der arbeitsspeicher",
        "ram nutzung anzeigen",
        "zeig den arbeitsspeicher verbrauch",
        "wie viel ram verbraucht das system",
        "arbeitsspeicher verbrauch anzeigen",
        "freier arbeitsspeicher",
        "wie viel ram habe ich noch",
        "ram checken",
        "prüfe den arbeitsspeicher",
        "wie viel hauptspeicher ist belegt",
        "hauptspeicher auslastung",
        "wie stark ist der arbeitsspeicher ausgelastet",
        "ram werte zeigen",
        "sag mir die ram auslastung",
        "arbeitsspeicher status",
        "wie ist der ram verbrauch",
        "wie viel vom arbeitsspeicher ist genutzt",
        "zeig mir den speicherverbrauch vom arbeitsspeicher",
        "hauptspeicher auslastung anzeigen",
        "wie viel hauptspeicher ist frei",
        "hauptspeicher verbrauch",
        "wie stark ist der hauptspeicher belegt",
        "arbeitsspeicher belegung",
        "wie belegt ist der arbeitsspeicher",
        "ram auslastung zeigen",
    ],
    "sys_disk": [
        "wie viel platz ist auf der festplatte",
        "festplatte auslastung",
        "wie voll ist die festplatte",
        "wie viel speicherplatz ist noch frei",
        "zeig mir den festplattenspeicher",
        "speicherplatz der festplatte anzeigen",
        "wie viel platz habe ich noch auf c",
        "laufwerk c auslastung",
        "wie viel gb sind auf der platte frei",
        "festplatten belegung anzeigen",
        "wie viel speicher ist auf der festplatte frei",
        "zeig den freien speicherplatz",
        "freier speicherplatz auf der platte",
        "wie groß ist die festplatte",
        "ist die platte voll",
        "prüfe den speicherplatz",
        "speicherplatz checken",
        "wie viel platz ist auf der ssd",
        "ssd auslastung anzeigen",
        "wie voll ist das laufwerk",
        "laufwerk belegung zeigen",
        "festplatte status",
        "wie viel speicherplatz bleibt noch",
        "zeig mir die festplatten auslastung",
        "wie viel platz ist auf der festplatte frei",
        "speicherplatz der platte anzeigen",
        "wie viel gb hat die festplatte noch frei",
        "ssd auslastung",
        "wie voll ist die ssd",
        "wie viel platz auf der ssd",
        "wie viel speicher hat die ssd noch",
        "festplatte belegung",
        "belegung der festplatte anzeigen",
        "laufwerk c speicher",
        "wie viel platz ist auf c frei",
        "speicherplatz auf laufwerk c",
        "die festplatte",
    ],
    "sys_summary": [
        "wie läuft der pc",
        "systemstatus anzeigen",
        "zeig mir den systemstatus",
        "wie geht es dem rechner",
        "gesamtauslastung anzeigen",
        "wie ist der stand des systems",
        "system übersicht",
        "gib mir eine systemübersicht",
        "wie läuft der computer",
        "status des systems anzeigen",
        "zeig mir eine übersicht über das system",
        "alles im grünen bereich",
        "system check",
        "systemzustand anzeigen",
        "zeig mir cpu ram und festplatte",
        "komplette systemübersicht",
        "wie läuft meine maschine",
        "generelle systeminfo",
        "kurzer systembericht bitte",
        "gib mir alle systemwerte",
        "wie steht das system da",
        "zeig den gesamtstatus des pcs",
        "pc zustand anzeigen",
        "wie performt der rechner gerade",
        "systemwerte anzeigen",
        "gib mir einen überblick über den pc",
        "gesamtauslastung des systems",
        "gesamtauslastung anzeigen lassen",
        "gesamtstatus des pcs",
        "gesamtstatus bitte",
        "gib mir einen überblick",
        "überblick über das system",
        "überblick bitte",
        "wie performt der pc",
        "wie performt der rechner",
        "wie performt das system gerade",
        "wie läuft das system",
        "zustand des pcs",
        "kurzer bericht über das system",
        "wie geht es dem pc gerade",
        "wie läuft der rechner so",
        "zustand des rechners",
        "wie steht es um den pc",
    ],
    "calc": [
        "was ist 12*4",
        "berechne 12*4",
        "rechne 3+4",
        "wie viel ist 15 geteilt durch 3",
        "berechne (2+3)*7",
        "was ergibt 2 hoch 10",
        "rechne 17,5 plus 2,5",
        "berechne 100 minus 45",
        "was ist 7 mal 8",
        "rechne 9*9",
        "wie viel ist 123 plus 456",
        "berechne 20 durch 4",
        "was ist 2 hoch 16",
        "rechne 5*(3+2)",
        "wie viel ist 1000 minus 275",
        "rechne 60 geteilt durch 5",
        "berechne 2+2*3",
        "was ist 144 durch 12",
        "rechne 0,5 plus 0,5",
        "wie viel sind 45 mal 32",
        "berechne 45 mal 32",
        "was ist -5 plus 10",
        "berechne 3 hoch 3",
        "was ist 2*(4+6)",
        "rechne 2*(4+6)",
        "wie viel ist 9 minus 16",
        "berechne 100 durch 7",
        "was ist 100 durch 7",
        "wie viel ist 5 hoch 2",
        "berechne 200 geteilt durch 8",
        "was macht 3 mal 3",
        "rechne bitte 22 plus 22",
        "kannst du 12*4 berechnen",
        "wie viel ist 60 mal 60",
        "rechne 3,5 mal 2",
        "was ist 1 plus 2 plus 3",
        "berechne (10-4) durch 2",
        "wie viel ist 7 hoch 2",
        "rechne 500 minus 125",
        "was ist 250 geteilt durch 4",
        "berechne 2 hoch 8",
        "rechne 99 plus 1",
        "wie viel ist 8 mal 7 minus 3",
        "berechne 50% von nichts",
        "was ist 10 hoch 3",
    ],
    "timer": [
        "stelle einen timer auf 10 minuten",
        "timer auf 5 minuten",
        "stell einen timer auf 30 sekunden",
        "timer 10 minuten",
        "setze einen timer auf 2 stunden",
        "stell mir einen timer auf 15 minuten",
        "wecker auf 10 minuten",
        "stelle einen wecker auf 20 minuten",
        "timer auf 90 sekunden",
        "starte einen timer über 5 minuten",
        "timer für 3 minuten",
        "5 minuten timer",
        "starte einen timer von 2 minuten",
        "erinner mich in 30 minuten",
        "erinnerung in 30 minuten",
        "erinner mich in 1 stunde",
        "stelle den timer auf 45 minuten",
        "timer auf eine stunde",
        "timer auf eine halbe stunde",
        "stelle einen timer auf eine halbe stunde",
        "wecker in 10 minuten",
        "timer über 60 sekunden",
        "stell einen kurzzeitwecker auf 5 minuten",
        "setze einen kurzzeitwecker auf 3 minuten",
        "timer 25 minuten bitte",
        "bitte einen timer auf 8 minuten",
        "kannst du einen timer auf 2 minuten stellen",
        "starte einen 5 minuten timer",
        "timer auf 120 sekunden",
        "wecker auf 30 minuten",
        "stelle einen timer auf 1 minute",
        "timer auf 2 minuten und 30 sekunden",
        "erinner mich in 2 stunden an die pause",
        "timer auf viertelstunde",
        "stelle einen timer auf eine viertelstunde",
        "timer auf 10 minuten und 20 sekunden",
        "mach mir einen timer auf 6 minuten",
        "timer 45 sekunden",
        "stell bitte einen wecker auf 5 minuten",
        "starte einen timer für 20 minuten",
        "erinner mich in einer viertelstunde",
        "wecker in einer halben stunde",
        "timer auf eine viertel stunde",
        "kurzzeitwecker auf eine halbe stunde",
        "viertel stunde timer",
        "halbe stunde timer",
        "erinner mich in einer halben stunde",
    ],
}

# ---------------------------------------------------------------------------
# Handwritten typo variants (category "typo"). The generator adds more
# deterministic random typos on top of these.
# ---------------------------------------------------------------------------

HANDWRITTEN_TYPOS: dict[str, list[str]] = {
    "open_app": [
        "öfne notepad",
        "öffn notepad",
        "öffne noteped",
        "starte notpad",
        "öffne den editr",
        "starte den broser",
        "öffne paintt",
        "öffne den taschenrechnr",
    ],
    "close_app": [
        "schliese notepad",
        "beende noteped",
        "mach notepad z",
        "beende firefoxx",
        "schließ den editr",
        "browser schliesen bitte",
        "beende wort",
        "mach excel zuu",
    ],
    "open_url": [
        "öffne wikepdeia",
        "öffne wikipeia",
        "geh auf gooogle",
        "öffne die webseite wikipeda",
        "öffne yutube",
        "mach gihub auf",
    ],
    "open_file": [
        "öffne die notzen",
        "öffne die datei notzen",
        "mach die notizen uaf",
        "öffne mein protokol",
        "öffne die rechnnug",
        "öffne das dokumnt",
    ],
    "open_folder": [
        "öffne die donloads",
        "öffne den ordner dokumnte",
        "mach den ordner downloads uaf",
        "öffne meine bilde",
        "öffne die dokumnte",
        "öffne den bilder odner",
    ],
    "find_file": [
        "finde die datei rechnnug",
        "suche rechnug",
        "wo ist die datei protokol",
        "suche die notzen",
        "finde das protokol",
        "such nach der rechnung",
    ],
    "sys_cpu": [
        "wi hoch ist die cpu auslastung",
        "cpu auslastug",
        "zeig mir die prozessorauslastug",
        "cpu auslastng anzeigen",
        "wie starc ist die cpu ausgelastet",
    ],
    "sys_ram": [
        "wi viel arbeitsspeicher ist belegt",
        "ram auslastug",
        "wie viel arm ist frei",
        "arbeitsmpeicher anzeigen",
        "wie voll ist der arbeitspeicher",
    ],
    "sys_disk": [
        "wie fol ist die festplatte",
        "festplatte auslastug",
        "wie viel platz ist auf der festplatt",
        "speicherplatz checkn",
        "wie viel speicherplatz ist noch fre",
    ],
    "sys_summary": [
        "wie lauft der pc",
        "systemstatus anzeigenn",
        "zeig mir den systemstatüs",
        "system übesicht",
        "wie geht es dem rechnr",
    ],
    "calc": [
        "berchne 12*4",
        "rechen 3+4",
        "was istt 12*4",
        "wieviel ist 15 geteilt durc 3",
        "berchne 100 minus 45",
        "was ergibt 2 hoh 10",
    ],
    "timer": [
        "stelle einen timr auf 10 minuten",
        "timer auf 5 miniten",
        "stell einen timer auf 2 stundn",
        "wecker auf 10 miniten",
        "erinner mich in 30 minueten",
        "starte einen timer über 5 minuuten",
    ],
}

# ---------------------------------------------------------------------------
# Out-of-domain / unsupported requests (category "unknown", label "unknown")
# ---------------------------------------------------------------------------

UNKNOWN_NORMAL: list[str] = [
    "hallo",
    "hi",
    "hey",
    "guten morgen",
    "guten abend",
    "gute nacht",
    "danke",
    "danke dir",
    "tschüss",
    "auf wiedersehen",
    "bis später",
    "wie geht es dir",
    "wie geht es dir heute",
    "wie geht's dir",
    "wie fühlst du dich",
    "wie geht es dir so",
    "wer bist du",
    "was kannst du",
    "hilf mir",
    "was ist los",
    "erzähl mir einen witz",
    "wie wird das wetter morgen",
    "was ist die hauptstadt von frankreich",
    "wer hat den kuchen gegessen",
    "schreib mir ein gedicht",
    "übersetze das für mich",
    "spiel musik",
    "mach die musik lauter",
    "musik leiser",
    "nächstes lied",
    "pause die musik",
    "mache einen screenshot",
    "drucke das dokument",
    "druck die datei",
    "starte den pc neu",
    "fahr den pc runter",
    "herunterfahren",
    "neustart",
    "schalte den rechner aus",
    "logge mich aus",
    "sperre den bildschirm",
    "ändere mein passwort",
    "lösche die datei notizen",
    "lösch den ordner downloads",
    "verschiebe die datei in den ordner",
    "kopiere die datei",
    "benenne die datei um",
    "komprimiere den ordner",
    "installiere spotify",
    "deinstalliere notepad",
    "aktualisiere windows",
    "update das system",
    "installiere updates",
    "schick eine e-mail an max",
    "schreibe eine mail",
    "beantworte die mail",
    "ruf mama an",
    "ich habe hunger",
    "was soll ich kochen",
    "erzähl mir was",
    "wie spät ist es",
    "welches datum ist heute",
    "wie ist der wechselkurs",
    "suche im internet nach katzenbildern",
    "mach das licht an",
    "heize die wohnung auf",
    "wie viel wiegt ein elefant",
    "wann ist ostern",
    "öffne die git historie im terminal",
    "schalte den drucker an",
    "verbinde mich mit dem wlan",
    "bluetooth aktivieren",
    "dunkler modus einschalten",
    "hintergrundbild ändern",
    "zeige versteckte dateien an",
    "leere den papierkorb",
    "lüftergeschwindigkeit erhöhen",
    # Destructive or modifying verbs are NOT supported: reject them all.
    "lösche die datei protokoll",
    "lösche die datei bericht",
    "lösch die datei rechnung",
    "lösch den ordner dokumente",
    "lösch die downloads",
    "entferne die datei notizen",
    "entferne den ordner bilder",
    "lösche alles",
    "lösch mir die tabelle",
    "vernichte die datei",
    "lösche die datei mit dem bericht",
    "formatiere die festplatte",
    "formatiere die platte",
    "formatiere laufwerk c",
    "formatier die ssd",
    "lösche den verlauf",
    "starte den rechner neu",
    "starte windows neu",
    "pc neu starten",
    "rechner neu starten",
    "windows neu starten",
    "fahr den rechner herunter",
    "mach einen neustart",
    "verschiebe die downloads",
    "komprimiere die downloads",
    "benenne die notizen um",
    "verschiebe die notizen in dokumente",
    "erstelle einen ordner",
    "erstelle eine neue datei",
    "erstelle einen ordner bilder",
    "schreib in die notizen",
    "bearbeite die notizen",
    "lese mir die notizen vor",
    "drucke die notizen",
    "beende alle programme",
    "schließe alles",
    "mach alle fenster zu",
    # Generic questions that must not be answered with an action.
    "was ist das für ein tag",
    "was ist deine meinung",
    "was kannst du alles",
    "was kannst du für mich tun",
    "was machst du gerade",
    "was soll ich heute anziehen",
    # Process/app management is out of scope.
    "welche apps sind installiert",
    "liste alle programme auf",
    "wie viele programme laufen gerade",
    "welche prozesse laufen",
    "biege um",  # complete nonsense
    "wie viele böden hat ein wolkenkratzer",
]

# ---------------------------------------------------------------------------
# Ambiguous requests (category "ambiguous", label "unknown").
# These must NOT be mapped to a single action; the assistant must ask back.
# ---------------------------------------------------------------------------

AMBIGUOUS_EXAMPLES: list[str] = [
    "öffne das",
    "mach auf",
    "stopp",
    "stop",
    "weiter",
    "nochmal",
    "öffne beides",
    "mach die datei und den ordner auf",
    "zeig es mir",
    "speicher anzeigen",
    "wie ist der speicher",
    "speicher status",
    "auslastung anzeigen",
    "alles anzeigen",
    "zeig mir alles",
    "status bitte",
    "öffne die sache da",
    "starte das programm da",
    "das da öffnen",
    "schließ mal was",
    "wo ist die datei",
    "wie läuft es",
    "alles ok",
    "ist alles gut",
    "was ist mit dem speicher",
    "öffne das dings",
    "starte mal das",
    "zeig mir das teil",
]

# ---------------------------------------------------------------------------
# Deterministic typo augmentation
# ---------------------------------------------------------------------------

VOWEL_NEIGHBORS = {
    "a": "e", "e": "i", "i": "e", "o": "u", "u": "o",
    "ä": "a", "ö": "o", "ü": "u",
}


def _apply_typo(rng: random.Random, text: str) -> str | None:
    """Apply one random, realistic typing mistake to a letter of ``text``."""
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    if len(letters) < 3:
        return None
    for _ in range(10):
        pos = rng.choice(letters)
        ch = text[pos]
        kind = rng.choice(["drop", "swap", "dup", "neighbor"])
        if kind == "drop" and len(text) > 3:
            return text[:pos] + text[pos + 1 :]
        if kind == "swap" and pos + 1 < len(text) and text[pos + 1].isalpha():
            return text[:pos] + text[pos + 1] + ch + text[pos + 2 :]
        if kind == "dup":
            return text[:pos] + ch + ch + text[pos + 1 :]
        if kind == "neighbor" and ch.lower() in VOWEL_NEIGHBORS:
            replacement = VOWEL_NEIGHBORS[ch.lower()]
            replacement = replacement.upper() if ch.isupper() else replacement
            return text[:pos] + replacement + text[pos + 1 :]
    return None


def augment_with_typos(
    examples: list[tuple[str, str]], seed: int, fraction: float
) -> list[tuple[str, str]]:
    """Return additional typo variants for a fraction of ``examples``.

    ``examples`` is a list of ``(label, text)`` tuples.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for label, text in examples:
        if rng.random() >= fraction:
            continue
        typo = _apply_typo(rng, text)
        if typo and typo != text:
            out.append((label, typo))
    return out


def build_rows(seed: int = 42, typo_fraction: float = 0.22) -> list[dict]:
    rows: list[dict] = []
    seen_texts: set[str] = set()

    def add(label: str, text: str, category: str) -> None:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty example text")
        if text in seen_texts:
            raise ValueError(f"duplicate example text: {text!r}")
        seen_texts.add(text)
        row = {
            "id": "", "text": text, "label": label,
            "category": category, "version": DATASET_VERSION,
        }
        rows.append(row)

    # Action intents: normal + handwritten typos + augmented typos.
    for label, examples in NORMAL_EXAMPLES.items():
        for text in examples:
            add(label, text, "normal")
        for text in HANDWRITTEN_TYPOS[label]:
            add(label, text, "typo")
        base = [(label, text) for text in examples]
        label_seed = seed + sum(ord(char) for char in label)
        augmented = augment_with_typos(base, seed=label_seed, fraction=typo_fraction)
        for _, typo in augmented:
            try:
                add(label, typo, "typo")
            except ValueError:
                pass  # augmentation collided with an existing example; skip it

    # Unknown and ambiguous requests share the "unknown" label.
    for text in UNKNOWN_NORMAL:
        add("unknown", text, "unknown")
    for text in AMBIGUOUS_EXAMPLES:
        add("unknown", text, "ambiguous")

    # Assign stable ids per label.
    counters: Counter = Counter()
    for row in rows:
        counters[row["label"]] += 1
        row["id"] = f"{row['label']}-{counters[row['label']]:04d}"
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__).resolve().parent.parent / "data"
                 / f"intent_dataset.v{DATASET_VERSION}.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--typo-fraction", type=float, default=0.22)
    args = parser.parse_args()

    rows = build_rows(seed=args.seed, typo_fraction=args.typo_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = Counter(row["label"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    print(f"Wrote {len(rows)} examples to {args.output}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<14} {count:>4}")
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:<10} {count:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
