import mido
import numpy as np

import createmidi
import miditoarray


def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)


SONG_LENGTH = 2000
NUM_TRACKS = 5
TEMPO = 100


outport = mido.open_output()

noteArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
velocityArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
onOffArray = np.random.randint(-1, 2, size=(NUM_TRACKS, SONG_LENGTH))

createmidi.createMidi(noteArray, velocityArray, onOffArray, int(round(60000000 / TEMPO)))

mid = mido.MidiFile('midi/new_song.mid')


#mid = MidiFile('mario.mid')

playMidi(mid)