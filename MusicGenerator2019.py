import mido

from mido import MidiFile
from mido.midifiles import MidiTrack
from mido import MetaMessage
from mido import Message

import time
import numpy
from numpy import *
import math
import random

from createarray2019 import *
from createmidi import *
from prettyprintarray import *


def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)


outport = mido.open_output()

noteArray = createArray()

#prettyPrintArray(noteArray)
    
createMidi(noteArray)

mid = MidiFile('midi/new_song.mid')


#mid = MidiFile('mario.mid')

playMidi(mid)