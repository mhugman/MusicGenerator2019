import mido
import numpy as np

import midiFunctions


def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)


SONG_LENGTH = 2000
NUM_TRACKS = 2
TEMPO = 80


outport = mido.open_output()

noteArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
velocityArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
onOffArray = np.random.randint(-1, 2, size=(NUM_TRACKS, SONG_LENGTH))

midiFunctions.createMidi(noteArray, velocityArray, onOffArray, int(round(60000000 / TEMPO)), "new_song")

noteArray_mario, velocityArray_mario, onOffArray_mario = midiFunctions.parseMidi(mido.MidiFile('midi/mario.mid'))

midiFunctions.createMidi(noteArray_mario, velocityArray_mario, onOffArray_mario, int(round(60000000 / TEMPO)), "new_mario")

#mid = mido.MidiFile('midi/new_song.mid')
mid = mido.MidiFile('midi/new_mario.mid')
#mid = MidiFile('mario.mid')

playMidi(mid)