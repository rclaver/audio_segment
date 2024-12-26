#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de TextGrid
@author: rafael
"""
import sys, os
import textgrids

def genera_textgrid(file):
   grid = textgrids.TextGrid(file)

   # Assume "syllables" is the name of the tier containing syllable information
   for syll in grid['syllables']:
      # Convert Praat to Unicode in the label
      label = syll.text.transcode()
      # Print label and syllable duration, CSV-like
      print('"{}";{}'.format(label, syll.dur))


if __name__ == "__main__":
   dir_font = 'txt-wav'

   with os.scandir(dir_font) as it:
      arxius = list(it)
      arxius.sort(key=lambda x: x.name)   #ordena els arxius per nom

   for arxiu in arxius:
      if arxiu.is_file() and arxiu.name[-3:]=="txt":
         genera_textgrid(f'{dir_font}/{arxiu.name}')
