## Autogenerador de TextGrid

# Definir el directorio de trabajo
sound_directory$ = "/home/rafael/projectes/TTS/alineació/txt-wav/"

# crear una lista con los nombres de los archivos
strings = Create Strings as file list: "list", sound_directory$ + "*.wav"
nFiles = Get number of strings

# Crea una tabla llamada "datos", con 0 filas y dos columnas: archivo y duracion
table_ID = Create Table with column names... datos 0 archivo duracion

writeInfoLine: "Arxius llegits"
appendInfoLine: "--------------"
tamany_maxim = 0

# Abre los archivos que tengan los nombres de la lista
for i to nFiles
	selectObject: strings
	filename$ = Get string: i
	appendInfoLine: filename$
	Read from file: sound_directory$ + filename$

	# se trata de hallar el archivo de tamaño máximo (en segundos)
	end = Get total duration
	if end > tamany_maxim
	   tamany_maxim = end
	endif

	# se llena la tabla con filas que tienen el nombre y la duración total de cada archivo.
	select table_ID
	Append row
	Set string value... i archivo 'filename$'
	Set numeric value... i duracion end

	# Por cada archivo .wav, crea un TextGrid con dos tiers
	# y uno de estos con 4 separaciones (boundaries) espaciados cada n ms

	# Esta parte crea un TextGrid con 4 separaciones por objeto Sound.
	#To TextGrid: "fono", ""
	#Insert boundary: 1, 0.1
	#Insert boundary: 1, 0.2
	#Insert boundary: 1, 0.3
	#Insert boundary: 1, 0.4
endfor
