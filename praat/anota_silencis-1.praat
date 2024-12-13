##
# Divideix un arxiu d'audio en segments de so i silenci
##

dir$ = "/home/rafael/projectes/TTS/alineació/praat/test/"
file$ = "A_casa_he_arribat"
sufix$ = "-1"

form Annotate sound files for silence
    sentence Input_audio_fn /home/rafael/projectes/TTS/alineació/praat/test/A_casa_he_arribat.wav
    sentence Output_audio_fn /home/rafael/projectes/TTS/alineació/praat/test/A_casa_he_arribat-1.TextGrid
    real Min_pitch_(Hz) 70
    real Time_step_(s) 0.0 (= auto)
    real Silence_threshold_(dB) -20.0
    real Min_silence_dur_(s) 0.1
    real Min_sound_dur_(s) 0.1
    sentence Silent_interval_label zona de silenci
    sentence Sounding_interval_label zona de so
endform

# Load audio file
sound = Read from file: input_audio_fn$
selectObject: sound

# Annotation
textgrid = To TextGrid (silences): min_pitch, time_step, silence_threshold, min_silence_dur, min_sound_dur, silent_interval_label$, sounding_interval_label$
selectObject: textgrid

# Output textgrid
Save as text file: output_audio_fn$

# Cleanup
selectObject: sound
Remove

selectObject: textgrid
Remove
