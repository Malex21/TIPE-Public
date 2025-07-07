# TIPE : Reconnaissance des notes jouées sur un instrument de musique

En écoutant en direct à partir du microphone de l'ordinateur, ce programme génère un fichier MIDI des notes jouées par l'instrument.
Ce projet a été effectué dans le cadre de l'épreuve de TIPE pour les classes préparatoires en filière scientifique.

Ce projet n'est pas à sa version finale, des améliorations peuvent encore être implémentées.

## Utilisation

Il faut exécuter le fichier Python, dans un environnement avec les libraires Numpy, PyAudio et MIDIUtil.

Durant les premières secondes, l'ordinateur détermine un bruit de fond moyen pour effectuer de la suppression de bruit.

Ensuite, une sortie texte est affichée lors de l'exécution; elle montre les notes détectées en direct à ce moment. Une sortie visuelle est aussi affichée, elle montre les fréquences détectées lors de l'exécution.

Pour arrêter le programme, il suffit de fermer la fenêtre de la sortie visuelle. Ensuite un fichier midiOutput.mid apparait dans le répertoire d'exécution.

