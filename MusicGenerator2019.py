import mido
import numpy as np

import midiFunctions


def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)


SONG_LENGTH = 256000
NUM_TRACKS = 5
TEMPO = 900

# Mario about 900 BPM


outport = mido.open_output()

noteArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
velocityArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
onOffArray = np.random.randint(-1, 2, size=(NUM_TRACKS, SONG_LENGTH))

print("noteArray: ", noteArray)


midiFunctions.createMidi(noteArray, velocityArray, onOffArray, int(round(60000000 / TEMPO)), "new_song")

noteArray_mario = np.zeros(noteArray.shape).astype("int")
velocityArray_mario = np.zeros(noteArray.shape).astype("int")
onOffArray_mario = np.zeros(noteArray.shape).astype("int")

#print("note Array shape: ", noteArray.shape)
#print("note Array mario shape: ", noteArray_mario.shape)

noteArray_mario, velocityArray_mario, onOffArray_mario = midiFunctions.parseMidi(noteArray_mario, velocityArray_mario, onOffArray_mario, mido.MidiFile('midi/mario.mid'))

print("noteArray_mario: ", noteArray_mario)

midiFunctions.createMidi(noteArray_mario, velocityArray_mario, onOffArray_mario, int(round(60000000 / TEMPO)), "new_mario")

#mid = mido.MidiFile('midi/new_song.mid')
mid = mido.MidiFile('midi/new_mario.mid')
#mid = MidiFile('mario.mid')

playMidi(mid)