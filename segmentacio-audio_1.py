#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
wav2vec 2.0 / HuBERT - Forced Alignment

torchaudio proporciona un acceso fácil a modelos previamente entrenados con etiquetas asociadas.
'''
import torch
import torchaudio

import re
from typing import List
import IPython
import matplotlib.pyplot as plt

from torchaudio.pipelines import MMS_FA as bundle

output_wav = "wav/transcr_audio"
source_vaw = "/home/rafael/projectes/TTS/sintetitzador/recording_scripts/data/"
arxiu_audio = f"{source_vaw}39_mostra.wav"
arxiu_text = f"{source_vaw}39_mostra.lab"

def version():
   print("torch version:\t\t", torch.__version__)
   print("torchaudio version:\t", torchaudio.__version__)
   print("torch.device:\t\t", device)

def normalitza_text(text):
   text = text.lower().replace("’", "'")
   text = re.sub("([^a-z' àèéíòóúç])", " ", text)
   text = re.sub(' +', ' ', text)
   return text.strip()

def get_text(arxiu):
   with open(arxiu, "r") as f:
      return normalitza_text(f.read())

'''
Utilizamos un modelo Wav2Vec2 que esté entrenado para ASR:
-torchaudio.pipelines.MMS_FA
'''
def generar_model_emissio():
   global model, aligner, tokenizer
   model = bundle.get_model()
   model.to(device)
   tokenizer = bundle.get_tokenizer()
   aligner = bundle.get_aligner()
   print(bundle.get_dict())

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
   global model, aligner, tokenizer
   with torch.inference_mode():
      emission, _ = model(waveform.to(device))
      token_spans = aligner(emission[0], tokenizer(transcript))
   return emission, token_spans

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()

def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate, autoplay=True)

'''

'''
def aliniar_transcripcio_audio():
   global waveform, token_spans, num_frames, transcript
   text_normalized = get_text(arxiu_text)
   waveform, sample_rate = torchaudio.load(arxiu_audio, frame_offset=int(bundle.sample_rate), num_frames=int(4.6 * bundle.sample_rate))
   assert sample_rate == bundle.sample_rate

   transcript = text_normalized.split()
   tokens = tokenizer(transcript)
   emission, token_spans = compute_alignments(waveform, transcript)
   num_frames = emission.size(1)

   #plot_alignments(waveform, token_spans, emission, transcript)
   print("Text:", text_normalized)
   audio = IPython.display.Audio(waveform, rate=sample_rate, autoplay=True)
   with open(f"{output_wav}.wav", "wb") as f:
      f.write(audio.data)


if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   torch.random.manual_seed(0)

   print("\nMatriu d'emissió")
   generar_model_emissio()

   print("\nAliniar la transcripcio a l'audio")
   aliniar_transcripcio_audio()

   for i in range(0,len(transcript)):
      audio = preview_word(waveform, token_spans[i], num_frames, transcript[i])
      with open(f"{output_wav}_{i}.wav", "wb") as f:
         f.write(audio.data)

