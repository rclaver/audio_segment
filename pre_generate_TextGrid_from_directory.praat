## Aquest script obre tots els fitxers d'un directori que comencen amb una cadena determinada.
## Crea un TextGrid per a cadascun dels fitxers de so i, a continuació,
## llança el fitxer de so i el TextGrid a l'editor perquè pugueu fer marques.
## Haureu d'executar aquest script una vegada per a cada paraula del vostre conjunt de dades.
## Es configura així de manera que tots els tokens d'una paraula determinada es marquen en successió,
## augmentant la uniformitat entre tokens. Per especificar la paraula en què treballareu, només heu d'executar
## l'script'. Apareixerà un formulari que us permetrà especificar el vostre directori i la paraula a treballar.
## Si no voleu fer-ho d'aquesta manera, només assegureu-vos de deixar el camp "Word" EN BLANC
## quan executeu l'script -- suprimiu qualsevol valor predeterminat que s'hi mostri.
## Llavors revisaràs tots els fitxers.

# Les quatre línies següents són un formulari que demana la ubicació del directori que voleu utilitzar.
# Aquest formulari especifica dues variables, "Directory" i "Word".  Més endavant en l'script,
# aquestes variables es poden anomenar "directory$" i "word$", ja que totes dues són variables de cadena.
# També especifiquen els valors per defecte per a aquestes variables -- "C:\My Data\" i "cat".  Edita
# aquestes línies si vols canviar els valors predeterminats, però assegura't' de no oblidar
# la barra invertida final al final del nom del directori.

form Enter directory and search string
	sentence Directory
	sentence Word
endform

# First we will make a list of all the sound files in your directory that begin with
# the word you specified in the form.  Note:  The next line is where you would change
# sound file extensions, such as replacing ".wav" with ".aiff".

Create Strings as file list... file-list 'directory$''word$'*.wav

# Now we will set up a "for" loop -- the loop will iterate once for every file in the list we just made.
# First we will query our list to see how many filenames are in it, and store that number using the
# variable "number_of_files".  That variable will then be used in setting up the for loop.

number_of_files = Get number of strings
for x from 1 to number_of_files

#    Now we will set up a string variable called "current_file$" and use it to store the first
#    filename from the list.

     select Strings file-list
     current_file$ = Get string... x

#    Now that we have the filename, we read in that file:

     Read from file... 'directory$''current_file$'

#    Now I am setting up a variable called "object_name$" that will have the name of the
#    sound object.  This is basically equivalent to subtracting the ".wav" or ".aiff" from
#    the filename.  This will be useful if I want to refer to the sound object later in the script.

     object_name$ = selected$ ("Sound")

#    Now I make a text grid for the current sound file.
#    Note that I am making only one tier on which labels can occur, and it will be named
#    "vowels".  You can have multiple tiers, each with its own name.  For example, I
#    could've made three tiers by saying  To TextGrid... "utterances words segments"

     To TextGrid... "enunciado palabra vocal"

#    Since we have just created a TextGrid, it is automatically selected (i.e., active).  We need
#    both the textgrid and the sound object to be selected together, so I am going to add the
#    sound object to the selection.  This is where removing the ".wav" extension comes in handy:

     plus Sound 'object_name$'

#    Now we want to throw those two objects -- the Sound object and the Textgrid object -- into the editor:

     Edit

#    Now, we will tell the script to pause.  This will allow the user to step in and enter the appropriate
#    marks using the mouse and keyboard.  Note that the user does not need to save the textgrid -- this is
#    built into the script later.  Just click on "continue" when you have made the marks that you want.

     pause  Mark your segments!

#    Now we will save the TextGrid object, so that the user doesn't have to do it for each file.
#    The first step is to de-select the sound object, leaving us with only the TextGrid selected:

     minus Sound 'object_name$'

#    Now we save the textgrid.  Note that the textgrid will have the same filename as the
#    sound file that it goes with, except that it will have the extension ".TextGrid" -- this is another
#    instance where removing the ".wav" comes in handy:

     Write to text file... 'directory$''object_name$'.TextGrid

#    We are now ready to end the for loop, and go on to the next file.  However, to conserve
#    memory, we will first remove the objects that we are through with.  I like to do this by
#    selecting all the objects in the list, then de-selecting any we will still be using, such as
#    our list of filenames.

     select all
     minus Strings file-list
     Remove

# This specifies the end of the loop:

endfor


# The next three lines just do some clean up, and display a message in the Info window
# letting you know when you've reached the end of the list.

select Strings file-list
Remove
print All files processed -- that was fun!

## written by Katherine Crosswhite
## crosswhi@ling.rochester.edu
