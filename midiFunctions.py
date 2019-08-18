### SongGenerator2019

import mido
import numpy as np
from mido import MidiFile
from mido.midifiles import MidiTrack
from mido import MetaMessage
from mido import Message

def parseMidi(mid): 

    # Initialize the note Array 
    noteArray = np.zeros((1,1,1), dtype=(int,3))
        
    # First build the note Array vertically (one row for each track, excluding the one
    # track we already accounted for in initializing the array
    # go through the messages and add up all the delta times of the messages, so we can 
    # figure out the maximum track length in clicks. This will let us us know how
    # far to build out the array horizontally
    
    maxTrackLength = 0
    for i, track in enumerate(mid.tracks[1:]):
        
        trackLength = 0
        for message in track:
            trackLength = trackLength + message.time
            
        if trackLength > maxTrackLength: 
            maxTrackLength = trackLength
        if i > 0: 
            noteArray = np.vstack((noteArray, [[[(0,0,0)]]]))
            
    #raise ValueError(noteArray)
            
    #prettyPrintArray(noteArray)
    
    # Build the array horizontally, by adding empty vertical vectors for the length
    # of the longest track
        
    emptyVector = np.zeros((noteArray.shape[0],1,1), dtype=(int,3))
    
    #raise ValueError(maxTrackLength)
    
    for x in range(maxTrackLength): 
        noteArray = np.hstack((noteArray, emptyVector))
    
    #print(noteArray.shape)   
    #prettyPrintArray(noteArray)
        
    
    for i, track in enumerate(mid.tracks[1:]):
    
        #print 'Track {}: {}'.format(i, track.name)
        
        currentNotes = []
        
        if track.name not in exclusions: 
            
            currentTime = 0
            for message in track:
                
                if message.type == "note_on" or message.type == "note_off":
                    prevTime = currentTime
                    currentTime = currentTime + message.time
                    
                    # previous notes are the same as (not yet updated) current notes, 
                    # but the third value is 0 instead of 1 (denoting the fact that 
                    # the note is sustained, rather than hit)
                    prevNotes = []
                    for x in currentNotes: 
                         prevNotes.append((x[0], x[1], 0))
                    
                    # Fill in all the values since the previous message, and up to 
                    # but not including the time of the current message, with the
                    # previous notes
                    if len(prevNotes) > 0:     
                        deltaTime = currentTime - prevTime
                        for j in range(0, deltaTime - 1):     
                            noteArray[i][currentTime - 1 - j] = prevNotes
                    
                    # update the current Notes being played with the information in the
                    # message
                    if message.type == "note_on": 
                                
                        currentNotes.append((message.note + 1, message.velocity,1))
                    
                    elif message.type == "note_off": 
                        for x in currentNotes: 
                            if x[0] == message.note + 1: 
                                 currentNotes.remove(x)
                        #currentNotes.remove((message.note + 1, message.velocity))   
                     
                    # fill in the value for this particular time with the current Notes
                    if len(currentNotes) > 0:    
                        try:  
                            noteArray[i][currentTime] = asarray(currentNotes)
                        except: 
                            raise ValueError(message, i, currentTime, currentNotes)
                        
    
    #raise ValueError(noteArray.shape[1])            
    return noteArray

def createMidi(noteArray, velocityArray, onOffArray, TEMPO, filename): 

    '''
    This function will take a 3D numpy note Array (e.g. parsed from a midi file), and turn it into 
    a midi file which can be played back. 
    '''

    with MidiFile() as mid:
    
        # add an empty first track (for testing)
    
        mid.add_track(name= str(0))
        track = mid.tracks[0]
            
        #track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=96, notated_32nd_notes_per_beat=8, time=0))
        track.append(MetaMessage('set_tempo', tempo=TEMPO, time=0))

        numMessages = 0 
    
        for i in range(noteArray.shape[0]): 
            
            mid.add_track(name= str(i + 1))
            track = mid.tracks[i + 1]
            
            track.append(Message('control_change', channel = i, control=0, value=0, time=0))

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

            #track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=96, notated_32nd_notes_per_beat=8, time=0))
            #track.append(MetaMessage('set_tempo', tempo=TEMPO, time=0))

            timeSinceLastMessage = 0


            for j in range(noteArray.shape[1]):

                messageGenerated = False

                noteVelocity = velocityArray[i,j]
                onOff = onOffArray[i,j]

                noteValue = noteArray[i,j]
                    
                    
                if onOff == 1: 
                    track.append(Message('note_on', channel = i, note= noteValue, velocity=noteVelocity, time=timeSinceLastMessage))
                    messageGenerated = True

                elif onOff == -1: 

                    track.append(Message('note_off', channel = i, note= noteValue, velocity=noteVelocity, time=timeSinceLastMessage))
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

        mid.save('midi/' + filename + '.mid')



