"""Generate the versioned MiniPCAI intent dataset.

The dataset is the single source of truth for training the intent classifier.
It is a versioned JSONL file (``data/intent_dataset.v<N>.jsonl``) where each row
has the following fields:

* ``id``        - stable unique identifier (``<label>-<counter>``)
* ``text``      - the German user utterance
* ``label``     - one of the 12 action intents or ``unknown``
* ``category``  - ``normal`` | ``typo`` | ``unknown`` | ``ambiguous``
* ``version``   - dataset version (must match the file name; the loader rejects
                  files whose name and rows disagree)

The version in the file name is also how the application finds the dataset: the
highest ``intent_dataset.v*.jsonl`` in ``data/`` is used, so bumping
``DATASET_VERSION`` below is all that is needed to publish a new dataset.

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
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/generate_dataset.py`
    sys.path.insert(0, str(REPO_ROOT))

DATASET_VERSION = 3

# ---------------------------------------------------------------------------
# Registry-driven synthesis
#
# The dataset must only teach targets that the registry can actually resolve:
# training on "starte word" while the registry has no `word` entry produces a
# confident classifier and a user-visible rejection. The generator therefore
# reads the shipped registry and
#   * keeps only handwritten examples that resolve to exactly one entry,
#   * synthesizes additional phrasings from the registered aliases,
#   * drops "unknown" examples that became resolvable (e.g. web search).
# ---------------------------------------------------------------------------

# Phrasings per registry section: templates use ``{alias}`` and ``{id}``.
_SYNTH_TEMPLATES: dict[str, tuple[str, ...]] = {
    "apps": (
        "öffne {alias}", "starte {alias}", "mach {alias} auf", "{alias} öffnen",
        "{alias} starten", "öffne bitte {alias}", "kannst du {alias} starten",
        "ich möchte {alias} öffnen", "starte {alias} bitte", "öffne mal {alias}",
    ),
    "files": (
        "öffne {alias}", "öffne die datei {alias}", "zeig mir {alias}",
        "öffne mir {alias}", "mach {alias} auf", "öffne die {alias}",
        "{alias} öffnen", "kannst du {alias} aufmachen",
    ),
    "folders": (
        "öffne {alias}", "öffne den ordner {alias}", "mach {alias} auf",
        "zeig mir {alias}", "öffne mir bitte {alias}", "{alias} öffnen",
        "geh in {alias}",
    ),
    "websites": (
        "öffne {alias}", "geh auf {alias}", "zeig mir {alias}",
        "öffne die webseite {alias}", "bring mich zu {alias}", "starte {alias}",
        "{alias} öffnen", "öffne mal {alias}",
    ),
}

_SYNTH_CLOSE_TEMPLATES: tuple[str, ...] = (
    "schließe {alias}", "beende {alias}", "mach {alias} zu", "{alias} schließen",
    "beende {alias} bitte", "stoppe {alias}", "{alias} beenden", "mach {alias} zu bitte",
)

# Queries for synthesized web searches (never a registered alias, so the query
# extractor cannot mistake them for a target).
_WEB_QUERY_POOL: tuple[str, ...] = (
    "katzenbilder", "wetter morgen", "rezept für pizza", "urlaub in italien",
    "python tutorial", "was ist eine ki", "bahnverbindung nach berlin",
    "gute filme 2026", "wechselkurs euro dollar", "symptome einer erkältung",
)

_WEB_SEARCH_TEMPLATES: tuple[str, ...] = (
    "suche nach {query}",
    "suche im internet nach {query}",
    "google nach {query}",
    "recherchiere {query}",
    "such mir {query}",
    "finde im internet {query}",
    "ich möchte etwas über {query} wissen",
    "kannst du nach {query} suchen",
    "websuche {query}",
    "schau im internet nach {query}",
)

_WEB_SEARCH_PROVIDER_TEMPLATES: tuple[str, ...] = (
    "{alias} nach {query}",
    "suche mit {alias} nach {query}",
    "{alias} suche nach {query}",
)

# Handwritten typo variants for the web-search intent.
WEB_SEARCH_TYPOS: tuple[str, ...] = (
    "suche im intenet nach katzenbildern",
    "google nach wetter mrogen",
    "recherchire rezept für pizza",
    "such nach urlaub in italin",
    "websuche nach pyton tutorial",
)

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
# Dataset v2 additions: further paraphrases, near-miss rejections and
# ambiguous requests. They improve generalization to phrasings that the
# handwritten v1 lists do not cover and teach the classifier sharper
# boundaries between similar intents (e.g. RAM "speicher" vs. disk
# "speicherplatz") and between supported and unsupported requests.
# ---------------------------------------------------------------------------

EXTRA_NORMAL_EXAMPLES: dict[str, list[str]] = {
    "open_app": [
        "ich würde gerne den rechner starten",
        "kannst du mir bitte den editor aufmachen",
        "öffne doch mal das malprogramm",
        "starte bitte das mailprogramm",
        "mach mir bitte den browser auf",
        "ich bräuchte den texteditor",
        "öffne firefox für mich",
        "editor starten",
        "notepad bitte öffnen",
        "könnten sie word starten",
        "bring den taschenrechner hoch",
        "ich möchte paint benutzen",
        "mach chrome mal auf",
        "öffne mir bitte firefox",
        "ich will den editor öffnen",
        "den browser starten bitte",
        "kannste notepad starten",
        "ich hätte gerne word geöffnet",
        "öffne das malprogramm bitte",
        "starte firefox mal",
        "rechner starten bitte",
        "öffne den editor für mich",
        "ich möchte den medienplayer starten",
        "bring mir bitte vlc hoch",
        "kannst du chrome starten",
        "öffne teams bitte",
        "starte das mailprogramm bitte",
        "ich bräuchte bitte den rechner",
        "mach den editor mal auf",
        "taschenrechner starten bitte",
        "öffne paint mal",
        "ich würde gerne firefox öffnen",
        "starte den browser mal",
        "kannst du bitte word öffnen",
        "öffne excel für mich",
        "mach mir excel auf",
        "ich möchte chrome starten",
        "starte outlook bitte",
        "öffne den editor mal",
        "bring bitte den browser hoch",
    ],
    "close_app": [
        "mach mal den editor zu bitte",
        "der browser soll beendet werden",
        "beende bitte das mailprogramm",
        "schließe firefox jetzt",
        "notepad kann zugehen",
        "beende das programm word",
        "mach den taschenrechner zu bitte",
        "stoppe chrome",
        "ich möchte den editor schließen",
        "beende vlc bitte sofort",
        "kannst du firefox beenden bitte",
        "mach chrome zu",
        "der editor soll zugehen",
        "schließe bitte den browser",
        "beende excel bitte",
        "word bitte beenden",
        "ich will notepad schließen",
        "mach den browser mal zu",
        "stoppe bitte den editor",
        "beende das mailprogramm",
        "kannste chrome zu machen",
        "schließ mal word",
        "beende firefox bitte sofort",
        "der rechner soll beendet werden",
        "mach bitte vlc zu",
        "ich möchte word beenden",
        "stoppe firefox",
        "beende bitte chrome",
        "schließe den editor bitte",
        "mach notepad mal zu",
    ],
    "open_url": [
        "mach mal wikipedia auf",
        "ich möchte auf youtube gehen",
        "zeig mir bitte die webseite github",
        "öffne die suchmaschine google",
        "geh bitte auf wikipedia",
        "kannst du youtube aufrufen",
        "bring mich zu google",
        "die webseite wikipedia bitte öffnen",
        "ruf die seite github auf",
        "ich will zu wikipedia",
        "öffne youtube bitte für mich",
        "mach google mal auf",
        "zeig mir die wikipedia seite",
        "ich möchte google öffnen",
        "geh auf youtube bitte",
        "kannst du die webseite wikipedia öffnen",
        "ruf bitte youtube auf",
        "öffne die seite github",
        "starte bitte die webseite wikipedia",
        "mach mir wikipedia auf",
        "ich würde gerne zu google gehen",
        "zeig mir bitte google",
        "wikipedia bitte aufrufen",
        "öffne die github seite mal",
        "geh zu wikipedia",
    ],
    "open_file": [
        "öffne bitte die datei rechnung",
        "zeig mir das protokoll",
        "ich möchte den bericht sehen",
        "mach die datei tabelle auf",
        "kannst du die präsentation öffnen",
        "die rechnung möchte ich öffnen",
        "öffne mir bitte die notizen",
        "datei protokoll anzeigen",
        "ich hätte gern die tabelle geöffnet",
        "mach den bericht mal auf",
        "zeig mir bitte die rechnung",
        "ich will die notizen öffnen",
        "die datei tabelle bitte öffnen",
        "öffne die präsentation mal",
        "kannst du mir den bericht zeigen",
        "bericht anzeigen bitte",
        "mach mir die tabelle auf",
        "ich möchte die präsentation sehen",
        "öffne das protokoll bitte",
        "die notizen möchte ich sehen",
        "zeig mir die datei rechnung",
        "kannste die notizen aufmachen",
        "rechnung bitte öffnen",
        "mach die präsentation mal auf",
        "ich bräuchte die datei protokoll",
        "öffne die tabelle für mich",
        "tabelle bitte anzeigen",
        "die datei bericht öffnen bitte",
        "zeig mir mal die notizen",
        "ich hätte gerne die rechnung geöffnet",
    ],
    "open_folder": [
        "öffne bitte den downloads ordner",
        "ich möchte in die dokumente",
        "zeig mir die bilder",
        "mach den ordner musik auf",
        "kannst du die downloads öffnen",
        "bilder ordner bitte anzeigen",
        "öffne den dokumente ordner mal",
        "ich will die downloads sehen",
        "den videos ordner öffnen bitte",
        "bring den bilder ordner hoch",
        "zeig mir bitte die downloads",
        "mach die dokumente mal auf",
        "ich möchte den bilder ordner öffnen",
        "downloads bitte anzeigen",
        "öffne den musik ordner",
        "kannst du mir die bilder zeigen",
        "der downloads ordner soll aufgehen",
        "mach mir die dokumente auf",
        "ich will in die bilder",
        "öffne die dokumente für mich",
        "videos ordner bitte öffnen",
        "zeig mir den musik ordner",
        "bilder bitte anzeigen",
        "den downloads ordner mal öffnen",
        "ich hätte gern die dokumente geöffnet",
        "mach den videos ordner mal auf",
        "öffne die musik mal",
        "kannste die downloads aufmachen",
        "dokumente ordner bitte anzeigen",
        "ich möchte die musik sehen",
    ],
    "find_file": [
        "suche bitte die datei rechnung",
        "wo finde ich das protokoll",
        "ich suche die tabelle",
        "kannst du die notizen finden",
        "finde bitte die präsentation",
        "wo ist mein bericht gespeichert",
        "durchsuche die dokumente nach rechnung",
        "ich möchte die datei tabelle finden",
        "such mir bitte den bericht",
        "finde die pdf datei mit der rechnung",
        "wo ist die datei tabelle bitte",
        "kannst du mir die rechnung suchen",
        "ich möchte den bericht finden",
        "suche die datei protokoll bitte",
        "finde bitte die notizen",
        "wo habe ich die tabelle gespeichert",
        "such die präsentation bitte",
        "ich suche die datei bericht",
        "kannst du das protokoll finden",
        "finde mir bitte die rechnung",
        "wo liegt die datei tabelle",
        "durchsuche bitte die downloads nach rechnung",
        "suche nach der datei protokoll",
        "ich möchte die präsentation suchen",
        "finde die notizen bitte",
        "wo ist die präsentation bitte",
        "kannste mir den bericht suchen",
        "such bitte die datei rechnung",
        "ich suche mein protokoll",
        "finde die datei mit der tabelle",
    ],
    "sys_cpu": [
        "wie stark ist der prozessor ausgelastet",
        "sag mir bitte die cpu last",
        "wie ist die cpu auslastung gerade",
        "prozessor auslastung bitte anzeigen",
        "wie viel prozent cpu werden gerade genutzt",
        "zeig mir die prozessorlast",
        "wie hoch ist die prozessorlast bitte",
        "cpu status bitte",
        "ist die cpu stark ausgelastet",
        "aktuelle cpu auslastung bitte",
        "wie ist die prozessor auslastung",
        "sag mir die cpu auslastung bitte",
        "cpu last bitte anzeigen",
        "wie stark ist die cpu ausgelastet bitte",
        "zeig mir bitte die cpu auslastung",
        "prozessorlast bitte zeigen",
        "wie viel last hat die cpu gerade",
        "cpu werte bitte",
        "ist der prozessor stark ausgelastet",
        "aktuelle prozessorlast bitte",
    ],
    "sys_ram": [
        "wie viel arbeitsspeicher wird gerade benutzt",
        "sag mir den ram verbrauch",
        "wie voll ist der ram bitte",
        "ram auslastung bitte zeigen",
        "wie viel speicher ist noch frei bitte",
        "zeig mir den speicherverbrauch",
        "ist der arbeitsspeicher voll",
        "wie viel hauptspeicher habe ich noch",
        "ram status bitte anzeigen",
        "wie stark ist der speicher belegt",
        "wie viel ram ist belegt bitte",
        "sag mir bitte wie viel ram frei ist",
        "speicherverbrauch bitte anzeigen",
        "wie viel arbeitsspeicher ist frei bitte",
        "zeig mir bitte den ram verbrauch",
        "ist der speicher voll",
        "wie viel speicher wird gerade benutzt",
        "ram belegung bitte",
        "wie stark ist der arbeitsspeicher belegt bitte",
        "freier speicher bitte anzeigen",
        "hauptspeicher status bitte",
        "wie viel ram habe ich noch bitte",
        "wie stark ist der ram belegt",
        "ist viel speicher belegt",
        "wie viel speicher ist belegt bitte",
    ],
    "sys_disk": [
        "wie viel speicherplatz ist auf der festplatte frei",
        "sag mir die festplatten belegung",
        "ist die festplatte bald voll",
        "wie viel gb sind auf c noch frei",
        "speicherplatz bitte anzeigen",
        "zeig mir den freien platz auf der platte",
        "wie groß ist die ssd bitte",
        "festplatten speicher bitte prüfen",
        "wie viel platz habe ich noch bitte",
        "laufwerk c auslastung bitte",
        "wie voll ist die festplatte bitte",
        "sag mir bitte wie viel speicherplatz frei ist",
        "festplatten belegung bitte zeigen",
        "wie viel speicher ist auf c frei",
        "zeig mir bitte die festplatten auslastung",
        "ssd speicher bitte anzeigen",
        "wie viel gb hat die festplatte noch",
        "freier speicherplatz bitte",
        "wie ist die festplatten auslastung",
        "wie viel speicher ist auf laufwerk c frei",
        "wie viel speicherplatz ist auf c frei",
        "speicher auf c bitte anzeigen",
        "wie viel ist auf c belegt",
        "c laufwerk speicher bitte",
        "wie viel speicher hat laufwerk c noch",
        "speicherplatz auf c bitte prüfen",
        "wie voll ist laufwerk c bitte",
    ],
    "sys_summary": [
        "wie geht es dem system gerade",
        "gib mir einen systembericht",
        "zeig mir alle systemwerte bitte",
        "wie steht der pc da",
        "system übersicht bitte",
        "ist das system in ordnung",
        "wie performt das system bitte",
        "kurzer statusbericht über den rechner bitte",
        "zeig mir cpu ram und festplatte bitte",
        "überblick über das system bitte",
        "wie läuft der rechner bitte",
        "sag mir den systemstatus",
        "systemstatus bitte anzeigen",
        "ist der pc in ordnung",
        "wie geht es dem pc jetzt so",
        "gib mir bitte eine systemübersicht",
        "zeig mir den pc status",
        "wie steht es um das system bitte",
        "systemzustand bitte",
        "kurzer systembericht",
    ],
    "calc": [
        "was ergibt 15 mal 4",
        "rechne bitte 99 geteilt durch 9",
        "wie viel ist 8 plus 9",
        "berechne 3 hoch 4 bitte",
        "was ist 50 minus 17",
        "kannst du 6*7 ausrechnen",
        "rechne 2,5 plus 3,5",
        "wie viel ergibt 12 durch 4",
        "was ist 100 plus 200 bitte",
        "berechne (4+5)*2",
        "was ist 7 mal 7",
        "rechne 1000 minus 1",
        "wie viel ist 3 mal 3 mal 3",
        "berechne bitte 2 hoch 5",
        "was ergibt 50 plus 50",
        "kannst du 12 plus 12 rechnen",
        "rechne 8 durch 2",
        "wie viel ist 20 mal 5",
        "was ist 99 minus 33 bitte",
        "berechne 6 hoch 2",
        "rechne bitte 15 plus 15",
        "wie viel ist 100 durch 10",
        "was ergibt 4 mal 25",
        "berechne 1 plus 1",
        "kannst du mir 5*5 sagen",
    ],
    "timer": [
        "stelle bitte einen timer auf 7 minuten",
        "timer auf 20 sekunden bitte",
        "erinnere mich in 45 minuten",
        "wecker auf 2 stunden bitte",
        "stelle einen timer auf 90 minuten",
        "timer für 15 minuten bitte",
        "erinnerung in 10 minuten bitte",
        "stelle einen kurzen timer auf 30 sekunden",
        "timer auf eine stunde bitte",
        "kannst du einen timer auf 5 minuten stellen",
        "stelle einen timer auf 3 minuten bitte",
        "wecker auf 60 sekunden bitte",
        "erinnere mich bitte in 20 minuten",
        "timer auf 25 minuten bitte",
        "setze bitte einen timer auf 10 minuten",
        "stelle einen wecker auf 15 minuten",
        "timer über 40 sekunden bitte",
        "erinnerung in 2 stunden bitte",
        "stelle mir bitte einen timer auf 8 minuten",
        "kurzzeitwecker auf 10 minuten bitte",
        "timer auf 5 minuten und 10 sekunden",
        "stelle einen timer auf 2 minuten bitte",
        "wecker in 30 minuten bitte",
        "kannst du mir einen wecker auf 5 minuten stellen",
        "timer auf 12 minuten bitte",
    ],
}

EXTRA_HANDWRITTEN_TYPOS: dict[str, list[str]] = {
    "open_app": [
        "ich würde gerne den rechne starten",
        "kannst du mir bitte den editr aufmachen",
        "öffne doch mal das malproramm",
        "starte bitte das mailproramm",
        "mach mir bitte den broweser auf",
        "öffne firefox für mich bitte",
        "notpad bitte öffnen",
        "rechnr starten bitte",
    ],
    "close_app": [
        "mach mal den editr zu bitte",
        "der broweser soll beendet werde",
        "beende bitte das mailproramm",
        "schließe firefox jetz",
        "stope chrome",
        "mach notepad mal z",
    ],
    "open_url": [
        "mach mal wikipdeia auf",
        "öffne die suchmaschiene google",
        "kannst du yutube aufrufen",
        "ruf die seite gihub auf",
        "geh zu wikpedia",
    ],
    "open_file": [
        "öffne bitte die datei rechnug",
        "zeig mir das protokol",
        "mach die datei tabelle uaf",
        "kannst du die präsentatoin öffnen",
        "tabelle bitte anzeigenn",
    ],
    "open_folder": [
        "öffne bitte den donloads ordner",
        "mach den ordner musik uaf",
        "kannst du die downlaods öffnen",
        "bilder ordner bitte anzeigenn",
        "öffne den musik odner",
    ],
    "find_file": [
        "suche bitte die datei rechnnug",
        "wo finde ich das protokol",
        "kannst du die notzen finden",
        "finde bitte die präsentatoin",
        "wo liegt die datei tabel",
    ],
    "sys_cpu": [
        "wie starc ist der prozessor ausgelastet",
        "sag mir bitte die cpu lst",
        "prozessor auslastung bite anzeigen",
        "cpu status bite",
    ],
    "sys_ram": [
        "wie viel arbeitspeicher ist noch frei bitte",
        "sag mir den ram verbaruch",
        "ram status bite anzeigen",
        "wie starc ist der speicher belegt",
    ],
    "sys_disk": [
        "wie viel speicherpltz ist auf der festplatte frei",
        "sag mir die festplatten belegun",
        "ist die festplatte bald vol",
        "speicherpltz bitte anzeigen",
    ],
    "sys_summary": [
        "wie geht es dem system gerde",
        "gib mir einen systembericnt",
        "system übersicht bite",
        "ist der pc in ordnung bitte",
    ],
    "calc": [
        "was ergibt 15 mal 4 bite",
        "rechne bitte 99 geteilt durc 9",
        "berechne 3 hcoh 4 bitte",
        "kannst du 6*7 ausrechenen",
    ],
    "timer": [
        "stelle bitte einen timr auf 7 minuten",
        "timer auf 20 sekunden bite",
        "erinnere mich in 45 miniten",
        "wecker auf 2 stundn bitte",
        "timer auf 25 minitten bitte",
    ],
}

# Near-miss requests that must NOT be executed. Many look similar to supported
# intents but ask for something the MVP deliberately does not do (percentages,
# square roots, currency conversion, unregistered apps, ...).
EXTRA_UNKNOWN: list[str] = [
    "was ist mit dem system los",
    "mach mal was",
    "zeig mir was",
    "starte irgendetwas",
    "beende irgendwas",
    "öffne alles",
    "wie hoch ist der speicher",
    "wie ist der stand",
    "sag mir die auslastung",
    "zeig die werte",
    "ist der pc ok",
    "was läuft gerade",
    "wie ist die lage",
    "mach was auf",
    "schließe irgendwas",
    "starte mal was",
    "suche was",
    "finde was",
    "wie viel prozent sind 50 von 200",
    "was ist die wurzel aus 16",
    "wie viel sind 100 euro in dollar",
    "rechne die prozent aus",
    "was ist der umsatzsteuersatz",
    "öffne meine emails",
    "öffne meinen kalender",
    "öffne die einstellungen",
    "öffne den task manager",
    "öffne den papierkorb",
    "öffne die energieeinstellungen",
    "öffne die fenster",
    "öffne die kamera",
    "öffne das netzwerk",
    "öffne die lautstärke",
    "ändere die auflösung",
    "ändere die sprache",
    "stelle die uhrzeit um",
    "aktualisiere die treiber",
    "leere den zwischenspeicher",
    "repariere die festplatte",
    "prüfe die festplatte auf fehler",
    "wie viele dateien habe ich",
    "wie groß sind meine dokumente",
    "zeig mir alle dateien an",
    "liste die programme auf",
    "welche apps laufen",
    "wie viele kerne hat die cpu",
    "wie viel watt verbraucht der pc",
    "wie heiß ist der prozessor",
    "zeig mir die temperatur",
    "wie alt ist der pc",
    "welche windows version habe ich",
    "zeig mir die ip adresse",
    "wie schnell ist mein internet",
    "mach einen speedtest",
    "verbinde mich mit dem internet",
    "schalte bluetooth aus",
    "aktiviere den flugmodus",
    "mach einen screenshot vom bildschirm",
    "nimm ein foto auf",
    "zeichne etwas auf",
    "übersetze den text ins englische",
    "fasse das dokument zusammen",
    "lese die nachrichten vor",
    "schreib eine email an anna",
    "beantworte die mail von max",
    "bestell mir eine pizza",
    "spiel ein lied ab",
    "nächstes video bitte",
    "erhöhe die lautstärke",
    "verringere die helligkeit",
    "aktiviere den dunklen modus",
    "leere den papierkorb bitte",
    "wer ist bundeskanzler",
    "wann kommt der weihnachtsmann",
    "ist heute ein feiertag",
    "was kostet ein flug nach mallorca",
    "sag mir die uhrzeit",
    "wie viele einwohner hat berlin",
    "wie wird das wetter übermorgen",
    # Unsupported verbs applied to registered targets: never execute these.
    "bearbeite die datei rechnung",
    "bearbeite das protokoll",
    "bearbeite die downloads",
    "lese mir die rechnung vor",
    "lese das protokoll vor",
    "lese mir die downloads vor",
    "schreibe in die rechnung",
    "schreibe etwas in die notizen",
    "kopiere die rechnung",
    "verschiebe die rechnung in die dokumente",
    "benenne die rechnung um",
    "drucke die rechnung",
    "drucke die downloads",
    "speichere die notizen",
    "sichere die datei rechnung",
    "sichere die dokumente",
    "entferne notepad",
    "entferne firefox",
    "repariere notepad",
    "installiere firefox",
    "aktualisiere firefox",
    "leere die downloads",
    "leere die dokumente",
    "lösche die downloads",
    "lösche die dokumente",
    "lösche die bilder",
    "lösche die rechnung",
    "wie groß ist die datei rechnung",
    "wie viele dateien sind in den downloads",
    "wie viele dateien sind in den dokumenten",
    "öffne die notizen zum bearbeiten",
    "zeig mir den inhalt der festplatte",
    "mach die downloads kleiner",
    "sortiere die downloads",
    "benenne die downloads um",
]

# Ambiguous requests: several intents or targets are plausible, so the only
# safe answer is to ask for clarification (trained towards ``unknown``).
EXTRA_AMBIGUOUS: list[str] = [
    "beende das programm",
    "öffne irgendwas",
    "wie ist die auslastung",
    "speicher bitte",
    "auslastung bitte",
    "system bitte",
    "mach das auf",
    "starte etwas",
    "beende etwas",
    "öffne etwas",
    "zeig mir etwas",
    "rechne etwas",
    "öffne mal was",
    "beende mal was",
    "suche mal was",
    "öffne die datei",
    "starte das programm",
    "beende die anwendung",
    "schließe die anwendung",
    "wie ist die speicherauslastung",
    "zeig mir die auslastung",
    "wie voll ist der speicher",
    "timer",
    "wecker",
    "rechner",
    "berechne etwas",
    "stelle einen timer",
    "erinnere mich",
    "finde die datei",
    "suche die datei",
    # Bare pronouns / placeholders instead of a concrete target name.
    "starte das mal",
    "öffne das mal",
    "mach das mal",
    "starte das da",
    "öffne das da",
    "das da starten",
    "das mal starten",
    "mach das da auf",
    "beende das mal",
    "schließe das mal",
    "starte das ding",
    "öffne das teil",
    "beende das teil",
    "mach das teil zu",
    # "Find a file" without saying which file.
    "finde eine datei",
    "suche eine datei",
    "finde die datei bitte",
    "suche die datei bitte",
    "ich suche eine datei",
    "finde mir eine datei",
    "suche nach einer datei",
    "wo ist eine datei",
    "finde einen ordner",
    "suche einen ordner",
    # Bare resource words without a concrete question.
    "speicher",
    "speicher anzeigen lassen",
    "wie ist der speicherstand",
    "speicherverbrauch",
    "ram bitte",
    "cpu bitte",
    "festplatte bitte",
    "system status bitte",
    "pc status",
    "rechner status",
    "auslastung",
    "speicherplatz",
    "wie ist die speicherbelegung",
]

# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def load_registry():
    """Load the packaged registry with the default security policy."""
    from minipcai.paths import packaged_registry_path
    from minipcai.policy import default_policy
    from minipcai.registry import Registry

    return Registry.load(packaged_registry_path(), policy=default_policy())


def section_for_label(label: str) -> str | None:
    from minipcai.intents import TARGETED_INTENTS

    return TARGETED_INTENTS.get(label)


def resolves_to_one(registry, text: str, section: str) -> bool:
    """True when exactly one registry entry of ``section`` matches ``text``."""
    return len(registry.find_matches(text, section)) == 1


def is_web_search(registry, text: str) -> bool:
    from minipcai.targets import TargetError, resolve_target

    try:
        resolve_target("web_search", text, registry)
        return True
    except TargetError:
        return False


def registry_driven_normal_examples(registry) -> dict[str, list[str]]:
    """Synthesize in-scope examples from the registered aliases."""
    examples: dict[str, list[str]] = {}
    section_to_label = {
        "apps": "open_app",
        "files": "open_file",
        "folders": "open_folder",
        "websites": "open_url",
    }
    for section, label in section_to_label.items():
        texts: list[str] = []
        for entry in registry.section(section):
            for alias in entry.aliases:
                for template in _SYNTH_TEMPLATES[section]:
                    texts.append(template.format(alias=alias, id=entry.id))
        examples[label] = texts
    close_texts: list[str] = []
    for entry in registry.section("apps"):
        for alias in entry.aliases:
            for template in _SYNTH_CLOSE_TEMPLATES:
                close_texts.append(template.format(alias=alias, id=entry.id))
    examples["close_app"] = close_texts

    search_texts: list[str] = []
    for query in _WEB_QUERY_POOL:
        for template in _WEB_SEARCH_TEMPLATES:
            search_texts.append(template.format(query=query))
    for entry in registry.section("searchers"):
        for alias in entry.aliases:
            for query in _WEB_QUERY_POOL[:4]:
                for template in _WEB_SEARCH_PROVIDER_TEMPLATES:
                    search_texts.append(template.format(alias=alias, query=query))
    examples["web_search"] = search_texts
    return examples


def filter_resolvable(
    examples: dict[str, list[str]], registry
) -> tuple[dict[str, list[str]], list[str]]:
    """Drop handwritten examples whose target is not in the registry."""
    kept: dict[str, list[str]] = {}
    dropped: list[str] = []
    for label, texts in examples.items():
        section = section_for_label(label)
        if section is None:
            kept[label] = list(texts)
            continue
        keep: list[str] = []
        for text in texts:
            if resolves_to_one(registry, text, section):
                keep.append(text)
            else:
                dropped.append(text)
        kept[label] = keep
    return kept, dropped


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


def merged_examples() -> tuple[
    dict[str, list[str]], dict[str, list[str]], list[str], list[str]
]:
    """Combine the handwritten v1 lists with the v2 additions.

    Returns ``(normal_examples, handwritten_typos, unknown, ambiguous)``.
    Raises :class:`ValueError` when an addition references an unknown label.
    """
    unknown_labels = set(EXTRA_NORMAL_EXAMPLES) - set(NORMAL_EXAMPLES)
    if unknown_labels:
        raise ValueError(f"unknown labels in EXTRA_NORMAL_EXAMPLES: {sorted(unknown_labels)}")
    unknown_typo_labels = set(EXTRA_HANDWRITTEN_TYPOS) - set(NORMAL_EXAMPLES)
    if unknown_typo_labels:
        raise ValueError(
            f"unknown labels in EXTRA_HANDWRITTEN_TYPOS: {sorted(unknown_typo_labels)}"
        )
    normal_examples = {
        label: list(examples) + EXTRA_NORMAL_EXAMPLES.get(label, [])
        for label, examples in NORMAL_EXAMPLES.items()
    }
    handwritten_typos = {
        label: list(HANDWRITTEN_TYPOS[label]) + EXTRA_HANDWRITTEN_TYPOS.get(label, [])
        for label in NORMAL_EXAMPLES
    }
    handwritten_typos["web_search"] = list(WEB_SEARCH_TYPOS)

    registry = load_registry()
    normal_examples, dropped = filter_resolvable(normal_examples, registry)
    handwritten_typos, dropped_typos = filter_resolvable(handwritten_typos, registry)
    _report_dropped(dropped + dropped_typos)

    # Add registry-synthesized phrasings (deduplicated, order preserving).
    synthesized = registry_driven_normal_examples(registry)
    for label, texts in synthesized.items():
        existing = set(normal_examples.get(label, []))
        merged = list(normal_examples.get(label, []))
        for text in texts:
            normalized = " ".join(text.split())
            if normalized not in existing:
                merged.append(normalized)
                existing.add(normalized)
        normal_examples[label] = merged

    handwritten_typos["web_search"] = list(WEB_SEARCH_TYPOS)
    unknown = [text for text in UNKNOWN_NORMAL + EXTRA_UNKNOWN
               if not is_web_search(registry, text)]
    ambiguous = list(AMBIGUOUS_EXAMPLES + EXTRA_AMBIGUOUS)
    for label in ("web_search",):
        if label not in normal_examples:
            raise ValueError(f"no examples for label {label!r}")
    return normal_examples, handwritten_typos, unknown, ambiguous


def _report_dropped(dropped: list[str]) -> None:
    if dropped and "--quiet" not in sys.argv:
        print(
            f"[generator] dropped {len(dropped)} example(s) whose target is not in "
            "the registry (dataset/registry agreement)"
        )


def build_rows(seed: int = 42, typo_fraction: float = 0.22) -> list[dict]:
    rows: list[dict] = []
    seen_texts: set[str] = set()
    normal_examples, handwritten_typos, unknown_examples, ambiguous_examples = (
        merged_examples()
    )

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
    for label, examples in normal_examples.items():
        for text in examples:
            add(label, text, "normal")
        for text in handwritten_typos.get(label, []):
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
    for text in unknown_examples:
        add("unknown", text, "unknown")
    for text in ambiguous_examples:
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
        default=(REPO_ROOT / "minipcai" / "data"
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
