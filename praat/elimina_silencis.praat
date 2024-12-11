###
# Script Praat per detectar i eliminar silencis
##

beginPause: "Detecció i eliminació de silencis"
   comment: "Directori d'arxius wav"
      sentence: "dir_wav", "/home/rafael/projectes/TTS/alineació/test/"
   comment: "Umbrals de duració"
      positive: "duration_silence", "0.3"
      positive: "duration_speech", "0.1"
   clicked = endPause: "OK", 1

#if dir_wav$ = ""
#	dir_wav$ = chooseDirectory$("Selecciona el directori d'audios i TextGrids")
#endif

Create Strings as file list: "list", dir_wav$ + "*.wav"
numberOfFiles = Get number of strings

# Desa el fitxer wav
Save as WAV file... /home/rafael/projectes/TTS/alineació/test/file.wav