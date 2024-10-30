#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Gravació d'audios breus a partir de frases curtes
'''
import pyaudio
import wave
import codecs
import random
import os
import sys; sys.path.append("/home/rafael/bin"); import colors as c

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1       # channels, must be one for forced alignment toolkit to work
RATE = 16000       # freqüència de mostreig (sample rate)
RECORD_SECONDS = 5 # nombre de segons de temps per poder dir la frase

arxiu_de_frases = 'data/frases.txt'
dir_result = "txt-wav"   #directori dels arxius generats

# recording function
def record(text, file_name):
   #os.system('clear')
   print(f"\n{c.CB_GRN}** Gravant **{c.C_NONE}")
   print(f"\n{c.CB_WHT}Llegeix en veu alta:{c.CB_YLW}", end=" "); print("{} ".format(text)); print(c.C_NONE, end="")

   p = pyaudio.PyAudio()
   stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

   frames = []
   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      data = stream.read(CHUNK)
      frames.append(data)

   stream.stop_stream()
   stream.close()
   p.terminate()

   wf = wave.open(file_name, 'wb')
   wf.setnchannels(CHANNELS)
   wf.setsampwidth(p.get_sample_size(FORMAT))
   wf.setframerate(RATE)
   wf.writeframes(b''.join(frames))
   wf.close()
   os.system('clear')

def main(nom, sentence_txt):
   sentence_set = codecs.open(sentence_txt, 'r', ).read().split('\n')
   random.shuffle(sentence_set)
   print(f"-\n{c.C_CYN}NOTA: es convenient redirigir la sortida d'errors: {c.C_YLW}genera_audios.py 2>/dev/null{c.C_NONE}\n-")
   print(f"{c.CB_WHT}Si estàs preparat, prem la tecla 'Retorn'{c.C_NONE}", end="")
   input()
   os.system('clear')
   for n in range(0, len(sentence_set)):
      if sentence_set[n]:
         record(f"{n}:\t{sentence_set[n]}", f"{dir_result}/{n}_{nom}.wav")
         outxt = open(f"{dir_result}/{n}_{nom}.txt", "w")
         outxt.write(sentence_set[n])
         outxt.close()

   print(f'\n{c.CB_GRN}** Fi de la gravació **{c.C_NONE}\n')

# inici
main('frase', arxiu_de_frases)
