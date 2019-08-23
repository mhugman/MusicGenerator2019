### SongGenerator2019

import mido
import numpy as np
from mido import MidiFile
from mido.midifiles import MidiTrack
from mido import MetaMessage
from mido import Message

import globalvars


SONG_LENGTH = globalvars.SONG_LENGTH
MAX_POLYPHONY = globalvars.MAX_POLYPHONY
NUM_MIDI_TRACKS = globalvars.NUM_MIDI_TRACKS
TEMPO = globalvars.TEMPO 

MAX_SUSTAIN = 200 # maximum time to sustain a note before it gets removed
MIN_LENGTH = 100


NUM_TRACKS = NUM_MIDI_TRACKS * MAX_POLYPHONY



'''
def findTimeMultiple(mid): 

    smallestTime = 1000000000000.

    for i, track in enumerate(mid.tracks): 

        for message in track: 


            if message.type == "note_on" or message.type == "note_off":

                print("message.time: ", message.time)

                if message.time > 0 and message.time < smallestTime: 

                    smallestTime = message.time

    timeMultiple = 1. / smallestTime

    return timeMultiple
'''

def parseMidi(mid): 
    
    # Exclude certain tracks from being parsed, such as percussion
    exclusions = ["Percussion"]

    

    noteArray = np.zeros((NUM_TRACKS, SONG_LENGTH)).astype("int")
    velocityArray = np.zeros((NUM_TRACKS, SONG_LENGTH)).astype("int")
    onOffArray = np.zeros((NUM_TRACKS, SONG_LENGTH)).astype("int")

    maxSongLength = noteArray.shape[1]

    #print("max Song Length: ", maxSongLength)
        
    
    for i, track in enumerate(mid.tracks):
    
        #print 'Track {}: {}'.format(i, track.name)
        
        currentNotes = []

        delQ = []

        timeQ = []
        
        if track.name not in exclusions: 
            
            currentTime = 0
            for message in track:

                #print("message: ", message)

                if currentTime >= maxSongLength - MIN_LENGTH: 

                    break
                
                if message.type == "note_on" or message.type == "note_off":
                    prevTime = currentTime

                    #print("prevTime: ", prevTime)
                    currentTime = currentTime + message.time

                    #print("currentTime: ", currentTime)
                    
                    # previous notes are the same as (not yet updated) current notes, 
                    # but the third value is 0 instead of 1 (denoting the fact that 
                    # the note is sustained, rather than hit)
                    prevNotes = []
                    for x in currentNotes: 
                         prevNotes.append((x[0], x[1], 0))
                    
                    #print("prevNotes: ", prevNotes)
                    
                    # Fill in all the values since the previous message, and up to 
                    # but not including the time of the current message, with the
                    # previous notes
                    if len(prevNotes) > 0:     
                        deltaTime = currentTime - prevTime
                        for j in range(0, deltaTime - 1): 
                            # Polyphony: add additional notes to extra tracks
                            for k in range(MAX_POLYPHONY): 
                                try:     
                                    noteArray[i + k * NUM_MIDI_TRACKS][currentTime - 1 - j] = prevNotes[k][0]
                                    #print("filling in this note: ", prevNotes[0][0])
                                    velocityArray[i + k * NUM_MIDI_TRACKS][currentTime - 1 - j] = prevNotes[k][1]
                                    #print("with this velocity: ", prevNotes[0][1])
                                    onOffArray[i + k * NUM_MIDI_TRACKS][currentTime - 1 - j] = prevNotes[k][2]
                                except: 
                                    pass
                    
                    # update the current Notes being played with the information in the
                    # message
                    if message.type == "note_on": 
                                
                        currentNotes.append((message.note + 1, message.velocity,1))

                        delQ.append((message.note + 1, message.velocity,1))
                        timeQ.append(MAX_SUSTAIN)


                        #print("currentNotes: ", currentNotes)

                        # fill in the value for this particular time with the current Notes
                        if len(currentNotes) > 0:  
                            for k in range(MAX_POLYPHONY): 
                                
                                if k < len(currentNotes) and currentNotes[k][0] == message.note + 1: 
                                    try:  
                                        noteArray[i + k * NUM_MIDI_TRACKS][currentTime] = np.asarray(currentNotes[k][0])
                                        velocityArray[i + k * NUM_MIDI_TRACKS][currentTime] = np.asarray(currentNotes[k][1])
                                        onOffArray[i + k * NUM_MIDI_TRACKS][currentTime] = np.asarray(currentNotes[k][2])
                                    except: 
                                        #raise ValueError(message, i, currentTime, currentNotes)
                                        pass


                    
                    elif message.type == "note_off": 
                        for x in currentNotes: 
                            if x[0] == message.note + 1: 
                                try: 
                                    x_idx = delQ.index(x)
                                    delQ.pop(x_idx)
                                    timeQ.pop(x_idx)
                                    currentNotes.remove(x)


                                except: 

                                    print("timeQ: ", timeQ)
                                    print("trying to remove ", x)


                            ## TO DO : add -1 to onOffArray in appropriate place


                                

                        #currentNotes.remove((message.note + 1, message.velocity))   

                    
                     
                    
                    
                    #print("noteArray: ", noteArray[:,:currentTime])
                    #print("velocityArray: ", velocityArray[:,:currentTime])
                

                
                timeQ = [x - 1 for x in timeQ]

                while timeQ.count(0) > 0 : 

                    zeroIdx = timeQ.index(0)
                    try: 
                        currentNotes.remove(delQ[zeroIdx])
                    except: 
                        print("currentNotes: ", currentNotes)
                        print("trying to delete: ", delQ[zeroIdx])
                    delQ.pop(zeroIdx)
                    timeQ.pop(zeroIdx)

                

                    


                        
    
    #raise ValueError(noteArray.shape[1])            
    return noteArray, velocityArray, onOffArray

def createMidi(noteArray, velocityArray, onOffArray, adjTempo, filename): 

    '''
    This function will take a 3D numpy note Array (e.g. parsed from a midi file), and turn it into 
    a midi file which can be played back. 
    '''

    with MidiFile() as mid:
    
        # add an empty first track (for testing)
    
        #mid.add_track(name= str(0))
        #track = mid.tracks[0]
            
        #track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=96, notated_32nd_notes_per_beat=8, time=0))
        

        numMessages = 0 
    
        for i in range(noteArray.shape[0]): 
            
            mid.add_track(name= str(i))
            track = mid.tracks[i]

            track.append(MetaMessage('set_tempo', tempo=adjTempo, time=0))
            
            #track.append(Message('control_change', channel = 1, control=0, value=0, time=0))

            # Grand Piano
            track.append(Message('program_change', channel = 1, program=0, time=0))
            '''
            if i >= 0 and i < 3: 
                # Grand Piano
                track.append(Message('program_change', channel = i, program=0, time=0))

            elif i == 3: 
                # Electric Guitar (clean)
                track.append(Message('program_change', channel = i, program=27, time=0))

            elif i == 4: 
                # Electric Bass (finger)
                track.append(Message('program_change', channel = i, program=33, time=0))
            elif i == 5: 
                # Glockenspiel
                track.append(Message('program_change', channel = i, program=9, time=0))
            else: 
                # Electric Piano
                track.append(Message('program_change', channel = i, program=4, time=0))
            '''

            #track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=96, notated_32nd_notes_per_beat=8, time=0))
            #track.append(MetaMessage('set_tempo', tempo=TEMPO, time=0))

            timeSinceLastMessage = 0


            for j in range(min(noteArray.shape[1], velocityArray.shape[1], onOffArray.shape[1])):

                messageGenerated = False

                noteVelocity = velocityArray[i,j]
                onOff = onOffArray[i,j]

                noteValue = noteArray[i,j]

                if noteValue < 0 : 

                    noteValue = 0 

                elif noteValue > 127: 

                    noteValue = 127

                if noteVelocity < 0 : 

                    noteVelocity = 0 

                elif noteVelocity > 127: 

                    noteVelocity = 127
                    
                    
                if onOff > 0.6: 

                    track.append(Message('note_on', channel = 1, note= noteValue, velocity=noteVelocity, time=timeSinceLastMessage))
                    messageGenerated = True

                elif onOff < -0.6: 
                     

                    track.append(Message('note_off', channel = 1, note= noteValue, velocity=noteVelocity, time=timeSinceLastMessage))
                    messageGenerated = True

                else: 
                    # onOff value is 0, don't do anything
                    pass


                # if a message was generated (either note_on or note_off), then reset the time
                # since last message to 0
                if messageGenerated == True: 
                    # the value here resets to 1 instead of 0, because its been at least one tick 
                    # since the last message

                    timeSinceLastMessage = 1
                else: 
                    # if there was no message, simply increment the time since the last message
                    timeSinceLastMessage = timeSinceLastMessage + 1

        mid.save("midi/" +  filename + '.mid')



