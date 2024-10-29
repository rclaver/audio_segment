#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
wav2vec 2.0 / HuBERT - Fine-tuned ASR

El proceso de alineación se ve así:
  1. Estimar la probabilidad de etiqueta cuadro por cuadro a partir de la forma de onda de audio
  2. Genere la matriz de enrejado que representa la probabilidad de que las etiquetas estén alineadas
     en el paso de tiempo.
  3. Encuentre la ruta más probable a partir de la matriz de enrejado.

Aquí, utilizamos el modelo torchaudiode Wav2Vec2 para la extracción de características acústicas.
'''

'''
torchaudio proporciona un acceso fácil a modelos previamente entrenados con etiquetas asociadas.
'''
import torch
import torchaudio

import re
import IPython
import matplotlib.pyplot as plt
from dataclasses import dataclass

bundle = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_ES
#d_dir_vaw = "tutorial-assets/"
#d_arxiu_audio = torchaudio.utils.download_asset(d_dir_vaw+"0_mostra.wav")
output_wav = "wav/audio_segment"
source_vaw = "/home/rafael/projectes/TTS/sintetitzador/recording_scripts/data/"
arxiu_audio = f"{source_vaw}39_mostra.wav"
arxiu_text = f"{source_vaw}39_mostra.lab"
up = False

def version():
   print("torch version:\t\t", torch.__version__)
   print("torchaudio version:\t", torchaudio.__version__)
   print("torch.device:\t\t", device)

def normalitza_text(text):
   text = text.replace("’", "'")
   if up:
      text = re.sub("([^A-Z' ÀÈÉÍÒÓÚÇ])", " ", text.upper())
   else:
      text = re.sub("([^a-z' àèéíòóúç])", " ", text.lower())
   text = re.sub(' +', ' ', text).strip()
   return text.replace(' ', '|')


def get_text(arxiu):
   f = open(arxiu, "r")
   text = f.read()
   f.close()
   return normalitza_text(text)

'''
El primer paso es generar la probabilidad de clase de etiqueta de cada cuadro de audio.
Podemos utilizar un modelo Wav2Vec2 que esté entrenado para ASR.
Aquí utilizamos torchaudio.pipelines
-WAV2VEC2_ASR_BASE_960H    . up=True
-VOXPOPULI_ASR_BASE_10K_ES . up=False
-WAV2VEC2_XLSR_300M
-WAV2VEC2_XLSR_2B
'''
def generar_matriu_emissio():
   global waveform, emission, labels, up
   up = False
   model = bundle.get_model().to(device)
   labels = bundle.get_labels()
   with torch.inference_mode():
       waveform, _ = torchaudio.load(arxiu_audio)
       emissions, _ = model(waveform.to(device))
       emissions = torch.log_softmax(emissions, dim=-1)

   emission = emissions[0].cpu().detach()
   print("Etiquetes:\n", labels)

def plot(mode="emission"):
   fig, ax = plt.subplots()
   if mode=="emission":
      img = ax.imshow(emission.T)
      ax.set_title("Classe de probabilitat per marc")
      ax.set_xlabel("Temps")
      ax.set_ylabel("Etiquetes")
   else:
      img = ax.imshow(trellis.T, origin="lower")
      ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
      ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3))
   fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
   fig.tight_layout()

'''
Generar probabilidad de alineación (enrejado)
A partir de la matriz de emisión, a continuación generamos el enrejado que representa la probabilidad
de que las etiquetas de transcripción aparezcan en cada período de tiempo.

Trellis es una matriz 2D con eje de tiempo y eje de etiquetas.
El eje de etiquetas representa la transcripción que estamos alineando.
A continuación, utilizamos t para denotar el índice en el eje del tiempo y j para indicar el índice
en el eje de la etiqueta.doyodoyo​representa la etiqueta en el índice de etiqueta.
cj​ representa la etiqueta en la etiqueta del indice j.

Para generar, la probabilidad del paso de tiempo t+1, observamos el enrejado desde el paso del tiempo t
y emisión en el paso de tiempo t+1.
Hay dos caminos para llegar a t+1 con la etiqueta cj+1.
​El primero es el caso en el que la etiqueta era cj+1 ​en t y no hubo ningún cambio de etiqueta desde t a t+1.
El otro caso es donde la etiqueta era cj en t y pasó a la siguiente etiqueta cj+1 ​en t+1.

Dado que buscamos las transiciones más probables, tomamos el camino más probable para el valor de k(t+1,j+1)​,
y eso es:
k(t+1,j+1)​=max(k(t,j)​p(t+1,cj+1​),k(t,j+1)​p(t+1,repeat))
dónde k representa su matriz de enrejado, y p(t,cj​) representa la probabilidad de etiqueta cj
​en el tiempo t. repeat ​​​​representa el token en blanco de la formulación CTC.
'''
def generar_transcripcio_matriu():
   # We enclose the transcript with space tokens, which represent SOS and EOS.
   global tokens, transcript
   transcript = get_text(arxiu_text)
   dictionary = {c: i for i, c in enumerate(labels)}
   tokens = [dictionary[c] for c in transcript]
   print("\ntranscipció de la matriu:\n", list(zip(transcript, tokens)))

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

'''
Encontrar el camino más probable (backtracking)
Una vez generado el enrejado lo recorreremos siguiendo los elementos con alta probabilidad.
Comenzaremos desde el último índice de etiqueta con el paso de tiempo de mayor probabilidad, luego,
retrocederemos en el tiempo, eligiendo la permanencia (cj→cj) o la transición (cj→cj+1​), en función
de la probabilidad posterior a la transición kt,jp(t+1,cj+1) o kt,j+1p(t+1,repeat).

La transición se realiza una vez que la etiqueta llega al principio.
La matriz de enrejado se utiliza para encontrar la ruta, pero para la probabilidad final de cada segmento,
tomamos la probabilidad cuadro por cuadro de la matriz de emisión.
'''
def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("The path found by backtracking")
    plt.tight_layout()

@dataclass
class Point:
   token_index: int
   time_index: int
   score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

'''
Segmentar el camino
Ahora esta ruta contiene repeticiones para las mismas etiquetas, así que vamos a fusionarlas para que quede cercana
a la transcripción original.
Al fusionar los puntos de ruta múltiples, simplemente tomamos la probabilidad promedio de los segmentos fusionados.
'''
# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

#Visualització de la segmentació
def plot_trellis_with_segments(trellis, segments, transcript):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ax2.set_title("Label probability with and without repetation")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-0.1, 1.1)
    fig.tight_layout()

'''
Fusionar los segmentos en palabras
El modelo Wav2Vec2 utiliza '|'como límite de palabra, por lo que fusionamos los segmentos antes de cada aparición de '|'.
Luego, finalmente, segmentamos el audio original en audio segmentado y los escuchamos para ver si la segmentación es correcta.
'''
# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

# Visualització de la fusió
def plot_alignments(trellis, segments, word_segments, waveform, sample_rate=bundle.sample_rate):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1)

    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvspan(word.start - 0.5, word.end - 0.5, edgecolor="white", facecolor="none")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    # The original waveform
    ratio = waveform.size(0) / sample_rate / trellis.size(0)
    ax2.specgram(waveform, Fs=sample_rate)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
        ax2.annotate(f"{word.score:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55), annotation_clip=False)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    fig.tight_layout()

'''
Generar audio de cada segmento
'''
def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=bundle.sample_rate, autoplay=True)



if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   torch.random.manual_seed(0)

   print("Matriu d'emissió")
   generar_matriu_emissio()
   plot('emission')

   #Generar probabilidad de alineación (enrejado)
   generar_transcripcio_matriu()

   print("\nConstrucció de la graella")
   trellis = get_trellis(emission, tokens)
   plot("trellis")

   print("\nBacktracking")
   path = backtrack(trellis, emission, tokens)
   #for p in path:
   #   print(p)
   plot_trellis_with_path(trellis, path)

   print("\nSegmentar el camí")
   segments = merge_repeats(path)
   #for seg in segments:
   #    print(seg)
   plot_trellis_with_segments(trellis, segments, transcript)

   print("\nFusionar els segments en paraules")
   word_segments = merge_words(segments)
   for word in word_segments:
       print(word)
   plot_alignments(trellis, segments, word_segments, waveform[0])

   # Desa els segment d'audio
   for i in range(0,len(word_segments)):
      audio = display_segment(i)
      with open(f"{output_wav}_{i}.wav", "wb") as f:
         f.write(audio.data)

