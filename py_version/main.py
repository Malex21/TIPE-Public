
from typing import Callable
from numpy.typing import NDArray
import pyaudio
from numpy.fft import rfft
from numpy import frombuffer, int16, arange, zeros, float64, empty, log2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from midiutil.MidiFile import MIDIFile


################################################################################################
#                                       Fonctions utilitaires                                  #
################################################################################################


def getAvgNoiseFreqProfile() -> NDArray:
    """Enregistre l'audio pendant RECORD_SECONDS secondes, et renvoie le profil fréquenciel moyenné sur cette durée

    Returns:
        NDArray[floating[Any]]: Tableau représentant le profil fréquenciel moyenné
    """

    nbIterations = int(RECORD_SECONDS / PERIOD)
    accumulated = zeros(CHUNK // 2 + 1, dtype=float64)

    for i in range(nbIterations):

        data = stream.read(CHUNK) 
        npdata = frombuffer(data, dtype=int16)

        freqs = abs(rfft(npdata) / CHUNK)
        accumulated += freqs

    return accumulated / nbIterations


def imax(arr) -> int:
    """Indice du maximum du tableau en argument"""
    return max(enumerate(arr), key=lambda x: x[1])[0]


def imaxSlice(arr, start, end) -> int:
    """Indice du maximum (entre l'indice start et l'indice end) du tableau en argument"""
    return start + imax(arr[start:end])


def roundToNearestMultiple(x, n) -> int:
    """Renvoie x arrondi au multiple de n le plus proche de x"""
    return int(n * round(x / n))


def keyToFreq(n) -> float:
    """Renvoie la fréquence correspondant à la n-ième touche du piano (la 49ième touche est A4 de fréquence 440Hz) (n à partir de 1)
    La réciproque de cette fonction est freqToKey
    """
    return 2 ** ((n - 49) / 12) * 440


def freqToKey(f) -> int:
    """Renvoie l'indice (à partir de 1) de la touche du piano correspondant à la fréquence en argument (la 49ième touche est A4 de fréquence 440Hz)
    La réciproque de cette fonction est keyToFreq
    """
    return int(12 * log2(f / 440) + 49)


def safeGraphSubstract(arr1, arr2) -> NDArray[float64]:
    """Soustrait arr2 à arr1. Si un des points du tableau devient négatif, on le met à zéro

    Args:
        arr1 (NDArray[float64]): Tableau d'où l'on soustrait
        arr2 (NDArray[float64]): Tableau que l'on soustrait

    Returns:
        NDArray[float64]: Différence (arr1 - arr2)
    """

    out = empty(len(arr1))

    for i in range(len(arr1)):
        out[i] = max(0, arr1[i] - arr2[i])

    return out


################################################################################################
#                                       Paramètres audio                                       #
################################################################################################


CHUNK = 9600
# taille d'un échantillon récupéré par le micro
# 1024 = 2**10 : Plus c'est grand, plus c'est précis. 2**9 ou inférieur ne fonctionne pas.
# 4410 permet d'avoir des frequency bins de taille 10Hz (si RATE = 44100) selon la formule bin = RATE/CHUNK

FORMAT = pyaudio.paInt16
CHANNELS = 1  # nombre de canaux audio
RATE = 48000  # dépend du micro : c'est le nombre de points récupérés par le micro en une seconde
RECORD_SECONDS = 5  # nombre de secondes d'enregistrement pour estimer le bruit de fond
KEY_NUMBER = 88     # nombre de touches de piano

binSize = RATE / CHUNK # dans un tableau en domaine fréquentiel, deux fréquences consécutives sont séparées d'une distance binSize
PERIOD = CHUNK / RATE  # intervalle de temps entre une prise d'échantillons et la prochaine
POINT_NUMBER = (CHUNK // 2) + 1  # rfft(arr) renvoie un tableau de taille (n // 2) + 1 si n = len(arr) pair (ici n = CHUNK)

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)


################################################################################################
#                             Constantes pour reconnaissance de notes                          #
################################################################################################


notes = [keyToFreq(n) for n in range(1, KEY_NUMBER + 1)]
# fréquences des touches de piano allant de la 1ère touche à la 88ième touche
notesNames = [
    "A0",
    "A#0",
    "B0",
    "C1",
    "C#1",
    "D1",
    "D#1",
    "E1",
    "F1",
    "F#1",
    "G1",
    "G#1",
    "A1",
    "A#1",
    "B1",
    "C2",
    "C#2",
    "D2",
    "D#2",
    "E2",
    "F2",
    "F#2",
    "G2",
    "G#2",
    "A2",
    "A#2",
    "B2",
    "C3",
    "C#3",
    "D3",
    "D#3",
    "E3",
    "F3",
    "F#3",
    "G3",
    "G#3",
    "A3",
    "A#3",
    "B3",
    "C4",
    "C#4",
    "D4",
    "D#4",
    "E4",
    "F4",
    "F#4",
    "G4",
    "G#4",
    "A4",
    "A#4",
    "B4",
    "C5",
    "C#5",
    "D5",
    "D#5",
    "E5",
    "F5",
    "F#5",
    "G5",
    "G#5",
    "A5",
    "A#5",
    "B5",
    "C6",
    "C#6",
    "D6",
    "D#6",
    "E6",
    "F6",
    "F#6",
    "G6",
    "G#6",
    "A6",
    "A#6",
    "B6",
    "C7",
    "C#7",
    "D7",
    "D#7",
    "E7",
    "F7",
    "F#7",
    "G7",
    "G#7",
    "A7",
    "A#7",
    "B7",
    "C8",
]

# noteToMidi[noteName] = hauteur midi correspondante
noteToMidi = {
    "A0": 21,
    "A#0": 22,
    "B0": 23,
    "C1": 24,
    "C#1": 25,
    "D1": 26,
    "D#1": 27,
    "E1": 28,
    "F1": 29,
    "F#1": 30,
    "G1": 31,
    "G#1": 32,
    "A1": 33,
    "A#1": 34,
    "B1": 35,
    "C2": 36,
    "C#2": 37,
    "D2": 38,
    "D#2": 39,
    "E2": 40,
    "F2": 41,
    "F#2": 42,
    "G2": 43,
    "G#2": 44,
    "A2": 45,
    "A#2": 46,
    "B2": 47,
    "C3": 48,
    "C#3": 49,
    "D3": 50,
    "D#3": 51,
    "E3": 52,
    "F3": 53,
    "F#3": 54,
    "G3": 55,
    "G#3": 56,
    "A3": 57,
    "A#3": 58,
    "B3": 59,
    "C4": 60,
    "C#4": 61,
    "D4": 62,
    "D#4": 63,
    "E4": 64,
    "F4": 65,
    "F#4": 66,
    "G4": 67,
    "G#4": 68,
    "A4": 69,
    "A#4": 70,
    "B4": 71,
    "C5": 72,
    "C#5": 73,
    "D5": 74,
    "D#5": 75,
    "E5": 76,
    "F5": 77,
    "F#5": 78,
    "G5": 79,
    "G#5": 80,
    "A5": 81,
    "A#5": 82,
    "B5": 83,
    "C6": 84,
    "C#6": 85,
    "D6": 86,
    "D#6": 87,
    "E6": 88,
    "F6": 89,
    "F#6": 90,
    "G6": 91,
    "G#6": 92,
    "A6": 93,
    "A#6": 94,
    "B6": 95,
    "C7": 96,
    "C#7": 97,
    "D7": 98,
    "D#7": 99,
    "E7": 100,
    "F7": 101,
    "F#7": 102,
    "G7": 103,
    "G#7": 104,
    "A7": 105,
    "A#7": 106,
    "B7": 107,
    "C8": 108,
}


freqReference = 440  # A4
dFreq = 10  # A4

minimalNoteLength = 0.2                                 # en secondes
noiseFreqProfile = getAvgNoiseFreqProfile()

threshold = 200  # u.a.   // en vue de retirer les fréquences et notes d'intensité trop faible (<= threshold)
noteThreshold = threshold
harmonicThreshold = threshold
dynamicThresholdCoefficient = 5  # 5: on supprime les présences <= 20% de la plus grande présence. 4: 25%, n: ((1/n) * 100)%

theoreticalModelPowerCoefficient = 1
harmonicRemovalStrenghCoefficient = 4


################################################################################################
#                                       Paramètres MIDI                                        #
################################################################################################


midiOutput = MIDIFile(numTracks=1, adjust_origin=True)

track = channel = 0
volume = 100


################################################################################################
#                                       Paramètres d'affichage                                 #
################################################################################################

maxAmplitude = 10000
maxFreq = 5000

fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(1, 1, 1)

freqArray = (arange(CHUNK) / PERIOD)[:POINT_NUMBER]

ax.set(
    xlim=[0, maxFreq],
    ylim=[0, maxAmplitude],
    xlabel="Freq (Hz)",
    ylabel="Amplitude (u.a.)",
)

noiseLine = ax.plot(
    freqArray, noiseFreqProfile, label="Recorded noise profile", color="red"
)

(unfilteredFreqLine,) = ax.plot(
    freqArray, [0] * len(freqArray), label="Unfiltered frequency spectrum", color="green"
)

(filteredFreqLine,) = ax.plot(
    freqArray, [0] * len(freqArray), label="Filtered frequency spectrum", color="blue"
)

noteLines = [ax.axvline(x, 0, 0) for x in notes]
# lignes verticales montrant l'amplitude détectée de chaque note

harmonicLines = [ax.axvline(x, 0, 0, color="r", alpha=0.5) for x in notes]
# lignes verticales correspondant aux harmoniques détectées de chaque note détectée

ax.legend()

################################################################################################
#                                       Fonctions générales                                    #
################################################################################################



def freqToIndex(freq) -> int:
    """Retourne l'index dans le tableau freqs correspondant à la fréquence freq en argument"""
    return int(freq / binSize)


def getMaximumInFreqInterval(freqY, freqMin, freqMax) -> float:
    """Détermine l'amplitude maximale du tableau des amplitudes freqY, entre la fréquence freqMin et freqMax

    Args:
        freqY (NDArray[float64]): Tableau des amplitudes des fréquences
        freqMin (float): Extrémité gauche de l'intervalle de fréquences à considérer
        freqMax (float): Extrémité droite de l'intervalle de fréquences à considérer

    Returns:
        float: Amplitude maximale dans l'intervalle en entrée
    """
    roundedFreqMin = roundToNearestMultiple(freqMin, binSize)
    roundedFreqMax = roundToNearestMultiple(freqMax, binSize)

    iFreqMin = freqToIndex(roundedFreqMin)
    iFreqMax = freqToIndex(roundedFreqMax)

    if iFreqMax == iFreqMin:
        iFreqMax += 1

    return freqY[imaxSlice(freqY, iFreqMin, iFreqMax)]


def arrToNoteDict(arr) -> dict[str, float]:
    """Convertit le tableau des amplitudes des notes (88 touches) en dictionnaire

    Args:
        arr (NDArray[float64]): Tableau des amplitudes des fréquences des 88 touches du piano

    Returns:
        dict[str, float]: Dictionnaire qui associe à chaque nom de note son amplitude correspondante
    """
    noteDict = {}

    for i, note in enumerate(notesNames):
        noteDict[note] = arr[i]

    return noteDict


def getNotePresenceArr(freqY) -> NDArray[float64]:
    """Obtient le tableau des amplitudes des 88 notes à partir de freqY

    Pour chaque note, on obtient son amplitude en prenant le maximum sur un petit intervalle (de taille 2 * dFreq * (freqNote / freqReference))

    Args:
        freqY (NDArray[float64])): Tableau des amplitudes des fréquences

    Returns:
        NDArray[float64]: Tableau des amplitudes associées aux fréquences des 88 touches du piano
    """
    presences = zeros(KEY_NUMBER, dtype=float64)

    for i, note in enumerate(notes):

        freqNote = roundToNearestMultiple(note, binSize)

        freqInterval = dFreq * (freqNote / freqReference)
        # precision parfaite : ~0.75Hz (car 27.5Hz / 440Hz * dFreq)
        # Ceci est un intervalle qui varie selon la fréquence

        # freqInterval = dFreq
        # Ceci est un intervalle fixe

        presences[i] = getMaximumInFreqInterval(
            freqY, freqNote - freqInterval, freqNote + freqInterval
        )

    return presences


def getNotePresenceDict(freqY) -> dict[str, float]:
    """Renvoie le dictionnaire qui à chaque nom de note associe son amplitude à partir de freqY (tableau d'amplitudes des fréquences)"""
    return arrToNoteDict(getNotePresenceArr(freqY))


def getMostPresentFrequency(freqs) -> float:
    """Renvoie la fréquence de plus forte amplitude dans freqs (tableau d'amplitude de fréquences en argument)"""
    return freqArray[imax(freqs)]


def drawNotePresence(presences) -> None:
    """Dessine en bleu sur le plot les lignes verticales qui représentes les amplitudes détectées des fréquences associées aux touches du piano"""

    for i, line in enumerate(noteLines):
        line.set_ydata([0, presences[i] / maxAmplitude])


def drawHarmonicPresence(harmonicPresences) -> None:
    """Dessine en rouge sur le plot les lignes verticales qui représentes les amplitudes détectées des fréquences associées aux harmoniques"""

    for i, line in enumerate(harmonicLines):  # TODO: Factoriser ?
        line.set_ydata([0, harmonicPresences[i] / maxAmplitude])


def purify(presences, harmonicPresences) -> list[int]:
    """Algorithme de suppression des harmoniques

    On regarde le tableau des fréquences filtrées à partir de la gauche.
    La première fréquence d'amplitude non négligeable (càd d'amplitude non nulle dans presences après filtrage) correspond nécessairement à une fondamentale.
    On vient de détecter une note jouée, on l'enregistre.
    On applique un modèle qui, à partir de la fréquence et de son amplitude, sachant que l'instrument est un piano, prédit les harmoniques qui viennent
    avec cette fondamentale.
    On soustrait ensuite à presences la fondamentale et ses harmoniques prédites. On obtient un nouveau tableau presences qui idéalement est le même mais
    comme si cette note détectée n'avait jamais été jouée.
    Puis on réapplique cet algorithme jusqu'à ce que presences soit nul (toutes les notes ont été détectées et retirées).

    On renvoie la liste des notes détectées.

    Args:
        presences (NDArray[float64]): Tableau des présences (amplitude des fréquences) des notes du piano
        harmonicPresences (NDArray[float64]): Tableau des présences des harmoniques détectées. Cette fonction remplit le tableau en place.

    Returns:
        list[int]: Numéros (à partir de 1) des touches de piano détectées comme jouées. Liste croissante
    """

    i = 0
    detectedKeys = []

    # noteThreshold = 50
    presences[presences < noteThreshold] = 0.0

    while i < KEY_NUMBER:

        if presences[i] > 0:

            detectedKeys.append(i + 1)

            predictedHarmonics = predictHarmonicsTheoretical(
                keyToFreq(i + 1), presences[i]
            )

            harmonicPresences += harmonicRemovalStrenghCoefficient * predictedHarmonics
            presences = safeGraphSubstract(presences, harmonicRemovalStrenghCoefficient * predictedHarmonics)
            presences[presences < noteThreshold] = 0.0

        i += 1

    return detectedKeys


"""
Pour attenuer la présences d'oscillations dans les amplitudes, on peut prendre une moyenne sur un court interval de temps
et afficher cette moyenne au lieu de la véritable oscillations. Ainsi quand ça descend, la descente ressemble bien à un explonetielle décroissante
"""


def predictHarmonicsTheoretical(f0, ampl0) -> NDArray[float64]:
    """Modèle de prédiction des harmoniques se basant sur un piano idéal

    Dans un piano idéal, les harmoniques décroissent selon A/n, où A est l'amplitude de la fondamental, et n est le rang de l'harmonique.
    Les amplitudes des harmoniques de rang impair sont nulles.

    Args:
        f0 (float): Fréquence de la fondamentale
        ampl0 (float): Amplitude de la fondamentale

    Returns:
        NDArray[float]: Tableau de taille 88 représentant la contribution harmonique de la fondamentale. Ne contient pas l'amplitude de la fondamentale.
                        TODO: Est-ce normal ?
    """
    harmonics = zeros(KEY_NUMBER, dtype=float64)
    i = 2
    k = theoreticalModelPowerCoefficient

    # f0 ne vaut jamais 400hz ?
    harmonicIndex = freqToKey(f0) - 1

    while harmonicIndex < KEY_NUMBER:

        harmonics[harmonicIndex] = ampl0 / (i**k)
        # -1 dans l'index car key est dans [|1, 89|]

        harmonicIndex = freqToKey(i * f0) - 1
        i += 1

    return harmonics


def saveMidiFile() -> None:
    """ Enregistre le fichier midi sur disque, dans le répertoire du programme (.), sous le nom "midiOutput.mid"
    """
    
    with open("midiOutput.mid", "wb") as outfile:
        midiOutput.writeFile(outfile)


def noteHandler() -> Callable[[int, set[str]], None]:
    """Renvoie une fonction permettant l'enregistrement correcte dans le fichier midiOutput des notes détectées au fil du temps
    """

    noteStart = dict()    # dict[note] -> start


    def updateSuccessiveNotes(iteration, detectedNotes) -> None:
        """ Retire les notes qui ne sont plus détectées et les enregistre dans le fichier midi avec leur instant de départ et leur durée.
            Les nouvelles notes détectées sont ajoutées à la liste des notes jouées.

            Args:
                iteration (int): Numéro du tour ("tick") de la boucle principale
                detectedNotes (set[str]): Notes détectées à cet instant du temps
        """

        for stoppedNote in noteStart.keys() - detectedNotes:

            start = noteStart.pop(stoppedNote)
            length = 2 * iteration * PERIOD - start            # le * PERIOD vient du fait qu'il faut reconvertir iteration en secondes (il y a PERIOD itérations par secondes)

            if length >= minimalNoteLength:                       
                midiOutput.addNote(track, channel, noteToMidi[stoppedNote], start, length, volume)

        for note in detectedNotes:

            if note not in noteStart.keys():       # si elle est présente au tour précédent
                noteStart[note] = 2 * iteration * PERIOD
        
    
    return updateSuccessiveNotes
            

def updateThreshold(presences) -> None:

    global threshold, noteThreshold, harmonicThreshold

    threshold = max(100, max(presences) / dynamicThresholdCoefficient)
    noteThreshold = threshold
    harmonicThreshold = threshold


def updateFreqs(iteration):
    """Organise le filtrage des fréquences et l'entretien du dessin du spectre fréquentiel
    Entre chaque itération de cette fonction, un laps de temps de PERIOD (s) s'écoule
    """

    global handleNotes, threshold

    data = stream.read(CHUNK)
    numpydata = frombuffer(data, dtype=int16)

    freqs = abs(rfft(numpydata) / CHUNK)  # de longueur len(numpydata) / 2 + 1 soit (CHUNK // 2) + 1

    freqsFiltered = safeGraphSubstract(freqs, noiseFreqProfile)
    freqsFiltered[freqsFiltered < threshold] = 0.0
    finalFreqs = freqsFiltered

    # unfilteredFreqLine.set_ydata(freqs)
    filteredFreqLine.set_ydata(finalFreqs)

    presences = getNotePresenceArr(finalFreqs)
    harmonicPresences = zeros(KEY_NUMBER, dtype=float64)

    # for f0, ampl0 in zip(notes, presences):
    #     harmonicPresences += predictHarmonicsTheoretical(f0, ampl0)

    noteDict = arrToNoteDict(presences)
    # harmonicDict = arrToNoteDict(harmonicPresences)

    presences[presences < noteThreshold] = 0.0
    # harmonicPresences[harmonicPresences < harmonicThreshold] = 0.0

    # detectedNotes = [note for note, ampl in noteDict.items() if ampl >= noteThreshold]
    # print(*detectedNotes, end=" | ")

    updateThreshold(presences)

    detectedNotesList = [notesNames[key - 1] for key in purify(presences, harmonicPresences)]  # pour imprimer
    print(*detectedNotesList)

    # detectedHarmonics = [note for note, ampl in harmonicDict.items() if ampl >= harmonicThreshold]
    # print(*detectedHarmonics)

    drawHarmonicPresence(harmonicPresences)  # rouge
    drawNotePresence(presences)  # bleu

    # print(getMostPresentFrequency(finalFreqs))
    # print(getMostPresentNote(finalFreqs))

    detectedNotes = set(detectedNotesList)
    handleNotes(iteration, detectedNotes)


    return unfilteredFreqLine, filteredFreqLine, *noteLines, *harmonicLines


def showFrequencies() -> None:

    try:
        ani = FuncAnimation(fig, updateFreqs, interval=10, blit=True)
        plt.show(block=True)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":

    handleNotes = noteHandler()
    showFrequencies()
    saveMidiFile()
