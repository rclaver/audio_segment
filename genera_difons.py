#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Generació de dífons
'''
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
import os

'''
Funció per dividir l'audio en dífons (segments)
'''
def dividir_en_difonos(audio, transcripcio, dir_sortida, seq, tasa_muestreo=22050):
   # Cargar el audio
   audio, sr = librosa.load(audio, sr=tasa_muestreo)

   # Aquí és on es defineixen les fronteres dels dífons. Cal millorar-lo amb l'alineació de text-audio
   duracio_total = librosa.get_duration(y=audio, sr=sr)
   num_difons = len(transcripcio) - 1
   duracio_difon = duracio_total / num_difons

   if not os.path.exists(dir_sortida):
      os.makedirs(dir_sortida)

   for i in range(num_difons):
      ini = int(i * duracio_difon * sr)
      fi = int((i + 1) * duracio_difon * sr)
      audio_difon = audio[ini:fi]

      # Desar el dífon como arxiu WAV
      arxiu_sortida = os.path.join(dir_sortida, f'difon_{seq}_{transcripcio[i]}-{transcripcio[i+1]}.wav').replace(" ", "¤")
      sf.write(arxiu_sortida, audio_difon, sr)



from praatio import textgrid
from pydub import AudioSegment

def extraer_difonos():
   # Cargar el archivo TextGrid
   textgrid_path = '/ruta/al/archivo.TextGrid'
   tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)

   # Cargar el audio
   audio_path = '/ruta/al/archivo.wav'
   audio = AudioSegment.from_wav(audio_path)

   # Extraer y guardar dífonos
   tier_name = 'phones'  # Nombre de la capa que contiene los fonemas
   output_dir = '/ruta/salida_difonos/'

   if not os.path.exists(output_dir):
      os.makedirs(output_dir)

   # Iterar sobre los intervalos de fonemas
   intervals = tg.tierDict[tier_name].entryList
   for i in range(len(intervals) - 1):
      inicio = int(intervals[i].start * 1000)  # Convertir a milisegundos
      fin = int(intervals[i + 1].end * 1000)

      # Extraer el dífono
      difono_audio = audio[inicio:fin]
      nombre_difono = f"{intervals[i].label}_{intervals[i + 1].label}.wav"

      difono_audio.export(os.path.join(output_dir, nombre_difono), format="wav")



if __name__ == "__main__":
   dir_font = 'txt-wav'
   dir_sortida = 'audios'

   with os.scandir(dir_font) as it:
      arxius = list(it)
      arxius.sort(key=lambda x: x.name)   #ordena els arxius per nom

   seq = 0
   for arxiu in arxius:
      if arxiu.is_file() and arxiu.name[-3:]=="wav":
         print(seq, arxiu.name)
         nom = arxiu.name[0:-4]
         with open(f'{dir_font}/{nom}.txt') as f: transcripcio = f.read()
         dividir_en_difonos(f'{dir_font}/{nom}.wav', transcripcio, dir_sortida, seq:=seq+1)

   #extraer_difonos()
