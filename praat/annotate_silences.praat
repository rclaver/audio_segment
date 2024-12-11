##
# Splits a file into sound and silence segments
##

dir$ = "/home/rafael/projectes/TTS/alineació/praat/test/"
file$ = "A_casa_he_arribat"

form Annotate sound files for silence
    sentence Input_audio_fn dir$ + file$ + ".wav"
    sentence Output_audio_fn dir$ + file$ + ".TextGrid"
    real Min_pitch_(Hz) 100
    real Time_step_(s) 0.0 (= auto)
    real Silence_threshold_(dB) -25.0
    real Min_silence_dur_(s) 0.1
    real Min_sound_dur_(s) 0.1
    sentence Silent_interval_label shh!
    sentence Sounding_interval_label sound
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
