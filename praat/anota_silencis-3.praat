###
# Script Praat per detectar silencis i parts de parla
##

dir$ = "/home/rafael/projectes/TTS/alineació/praat/test/"
sufix$ = "-3"

beginPause: "Detecció de silencis i parla"
   comment: "Directori d'arxius wav"
      sentence: "dir_wav", dir$
   comment: "Umbrals de duració"
      positive: "duration_silence", "0.1"
      positive: "duration_speech", "0.1"
   clicked = endPause: "OK", 1

#if dir_wav$ = ""
#	dir_wav$ = chooseDirectory$("Selecciona el directori d'audios i TextGrids")
#endif

Create Strings as file list: "list", dir_wav$ + "*.wav"
numberOfFiles = Get number of strings

for i from 1 to numberOfFiles
	select Strings list
   	fileName$ = Get string: i
   	appendInfoLine: fileName$

	Read from file: dir_wav$+fileName$
	soundname$ = selected$ ("Sound")
	selectObject: "Sound 'soundname$'"
	#select Sound 'soundname$'
	noprogress To Pitch (cc): 0.0, 50, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14,600
	q1_f0 = Get quantile: 0, 0, 0.05, "Hertz"
	q3_f0 = Get quantile: 0, 0, 0.95, "Hertz"
	Remove

	selectObject: "Sound 'soundname$'"
	To Intensity: q1_f0, 0.005, "yes"
	q1_Int = Get quantile: 0, 0, 0.05
	q3_Int = Get quantile: 0, 0, 0.95
	intensitySD = Get standard deviation: 0, 0
	silenceThreshold = (q3_Int - q1_Int) - (intensitySD/2)
	silenceThreshold = - silenceThreshold
	Remove

	selectObject: "Sound 'soundname$'"
	textgrid = To TextGrid (silences): q1_f0, 0.005, silenceThreshold, duration_silence, duration_speech, "zona silenci", "zona de so"
	Write to text file: dir_wav$ + soundname$ + sufix$ + ".TextGrid"
	Remove

	selectObject: "Sound 'soundname$'"
	Remove
endfor

appendInfoLine: "-------------"
appendInfoLine: "--- Final ---"
