##
# Divideix un arxiu d'audio en segments de so i silenci
##

dir$ = "/home/rafael/projectes/TTS/alineació/praat/test/"
file$ = "A_casa_he_arribat"
sufix$ = "-2"
path$ = dir$ + file$
logfile$ = path$ + sufix$ + ".log"

start = 0.0
finish = 0.0
time_step = 0.0

# llindar d'un silenci
llindar_sil = -20
# duració mínima d'un silenci
dur_min_sil = 0.1
# duració mínima d'un so
dur_min_so = 0.1
# etiqueta d'un silenci
etq_sil$ = "zona de silenci"
# etiqueta d'un so
etq_so$ = "zona audio"

deleteFile (logfile$)

sound = Read from file: path$ + ".wav"
selectObject: sound

# Proceso de reducción de ruido y generación del audio modificado
# (no es necesario para la detección de las zonas de silencio)
#
   #Reduce noise: start, finish, 0.025, 80.0, 800.0, 10.0, -60.0, "spectral-subtraction"
   #Save as WAV file: path$ + "_denoised.wav"
   #selectObject: sound
   #Remove
#
   #sound = Read from file: path$ + "_denoised.wav"
   #selectObject: sound

finish = Get total duration

View & Edit
   editor: sound
      Show all
      Select: start, finish
      min_pitch = Get minimum pitch
      max_pitch = Get maximum pitch
      pitch_list$ = Pitch listing
   endeditor

# Annotation
textgrid = To TextGrid (silences): min_pitch, time_step, llindar_sil, dur_min_sil, dur_min_so, etq_sil$, etq_so$
selectObject: textgrid

# Output textgrid
Save as text file: path$ + sufix$ + ".TextGrid"

# log
appendFileLine (logfile$, "Paràmetres")
appendFileLine (logfile$, "  inici audio (segons) = ", start)
appendFileLine (logfile$, "  final audio (segons) = ", finish)
appendFileLine (logfile$, "  time_step = ", time_step)
appendFileLine (logfile$, "  llindar silenci (dB) = ", llindar_sil)
appendFileLine (logfile$, "  duracio_mínima_silenci (s) = ", dur_min_sil)
appendFileLine (logfile$, "  duracio_mínima_so      (s) = ", dur_min_so)
appendFileLine (logfile$, "  etiqueta_silenci = " + etq_sil$)
appendFileLine (logfile$, "  etiqueta_so      = " + etq_so$)
appendFileLine (logfile$, "Tons (freqüències / pitch)")
appendFileLine (logfile$, "  to mínim (Hz) = ", min_pitch)
appendFileLine (logfile$, "  to màxim (Hz) = ", max_pitch)
appendFileLine (logfile$, "  llistat de tons:")
appendFileLine (logfile$, pitch_list$)

# Cleanup
selectObject: sound
Remove

selectObject: textgrid
Remove
