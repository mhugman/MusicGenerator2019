import mido
import numpy as np
import torch
import math
import sys
import datetime

import midiFunctions
import globalvars

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10000)


############ GLOBAL PARAMETERS ######################

SONG_LENGTH = globalvars.SONG_LENGTH
MAX_POLYPHONY = globalvars.MAX_POLYPHONY
NUM_MIDI_TRACKS = globalvars.NUM_MIDI_TRACKS
TEMPO = globalvars.TEMPO

NUM_TRACKS = NUM_MIDI_TRACKS * MAX_POLYPHONY

LEARN_NOTES = True
LEARN_VEL = False
LEARN_ONOFF = True


H = 600 # dimensions for hidden layer
ITERATIONS = 30000
THRESHOLD = 1.0
LEARNING_RATE_NOTE = 1e-4 # 1e-4 for mario
LEARNING_RATE_VEL = 1e-5 # 1e-4 for mario
LEARNING_RATE_ONOFF = 1e-7

alpha = 0.999999

FILEPRE = "generated_midi/emulate_moonlight"
FILEPOST = "fixedparser"

FILESOURCE = "moonlight"

# Calculate global parameter for square Matrix

D = 0 

while D ** 2 < SONG_LENGTH * NUM_TRACKS: 

    D += 1

#####################################################

def toSquareArray(rectArray): 

    numChunks = int(math.ceil((SONG_LENGTH / NUM_TRACKS)))

    
    #print("numChunks: ", numChunks)
    #print("D: ", D)
    
    squareArray = np.zeros((D, D)).astype("int")

    numRowChunks = int(math.floor((D / NUM_TRACKS)))
    numColChunks = int(math.floor((D / NUM_TRACKS)))

    #print("numRowChunks: ", numRowChunks)

    chunk = 0

    for i in range(numRowChunks - 1): 


        for j in range(numColChunks - 1): 

            #print("i: ", i)
            #print("j: ", j)

            #print("row copying in square array from: ", NUM_TRACKS * i, "to ", NUM_TRACKS * i + NUM_TRACKS)
            #print("col copying in square array from: ", NUM_TRACKS * j, "to ", NUM_TRACKS * j + NUM_TRACKS)

            #print("rect source col from ", NUM_TRACKS * chunk, "to ", NUM_TRACKS * chunk + NUM_TRACKS)

            #print("source: ", rectArray[:, NUM_TRACKS * chunk : NUM_TRACKS * chunk + NUM_TRACKS])
            
            try: 
                squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS] = rectArray[:, NUM_TRACKS * chunk : NUM_TRACKS * chunk + NUM_TRACKS]
            except: 

                print("size mismatch to square")
            
            #print("destination: ", squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS] )

            chunk += 1

    return squareArray

def toRectArray(squareArray):

    numChunks = int(math.ceil((SONG_LENGTH / NUM_TRACKS)))
    
    #print("numChunks: ", numChunks)
    #print("D: ", D)
    
    rectArray = np.zeros((NUM_TRACKS, numChunks * NUM_TRACKS)).astype("int")

    numRowChunks = int(math.floor((D / NUM_TRACKS)))
    numColChunks = int(math.floor((D / NUM_TRACKS)))

    #print("numColChunks: ", numColChunks)

    chunk = 0

    for i in range(numRowChunks - 1): 


        for j in range(numColChunks - 1): 

            #print("i: ", i)
            #print("j: ", j)

            #print("row source square array from: ", NUM_TRACKS * i, "to ", NUM_TRACKS * i + NUM_TRACKS)
            #print("col source square array from: ", NUM_TRACKS * j, "to ", NUM_TRACKS * j + NUM_TRACKS)

            #print("rect destination col from ", NUM_TRACKS * chunk, "to ", NUM_TRACKS * chunk + NUM_TRACKS)

            #print("source: ", squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS])

            try: 
                rectArray[:, NUM_TRACKS * chunk : NUM_TRACKS * chunk + NUM_TRACKS] = squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS]
            except: 
                print("size mismatch to rect")

            #print(rectArray[:,:NUM_TRACKS * chunk + NUM_TRACKS])

            chunk += 1

    return rectArray



def playMidi(mid): 
    
    for message in mid.play():
        #print(message)
        outport.send(message)



outport = mido.open_output()

noteArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
velocityArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
onOffArray = np.random.randint(-1, 2, size=(NUM_TRACKS, SONG_LENGTH))

noteArray_square = toSquareArray(noteArray)

#midiFunctions.createMidi(noteArray, velocityArray, onOffArray, int(round(60000000 / TEMPO)), "new_song")

noteArray_source, velocityArray_source, onOffArray_source = midiFunctions.parseMidi(mido.MidiFile('midi/' + FILESOURCE + '.mid'))

#print("noteArray_source: ", noteArray_source)
#print("velocityArray_source: ", velocityArray_source)
#print("onOffArray_source: ", onOffArray_source)

noteArray_source_square = toSquareArray(noteArray_source)

velocityArray_source_square = toSquareArray(velocityArray_source)

onOffArray_source_square = toSquareArray(onOffArray_source)

midiFunctions.createMidi(noteArray_source, velocityArray_source, onOffArray_source, int(round(60000000 / TEMPO)), "parsed_" + FILESOURCE)

#mid = mido.MidiFile('midi/parsed_' + FILESOURCE + '.mid')

#playMidi(mid)

#raise ValueError(234234)



############ DEEP LEARNING ########################

# source: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

#print("onOffArray_source:", onOffArray_source)


# Create random Tensors to hold inputs and outputs
#x = torch.from_numpy(noteArray_square.astype("float")).float()

x_note = torch.randn(D, D)
x_vel = torch.randn(D, D)
x_onOff = torch.randn(D, D)


y_note = torch.from_numpy(noteArray_source_square.astype("float")).float()
y_vel = torch.from_numpy(velocityArray_source_square.astype("float")).float()
y_onOff = torch.from_numpy(onOffArray_source_square.astype("float")).float()

y_note = y_note * (1./128)
y_vel = y_vel * (1./128)
y_onOff = y_onOff * 1000


model_note = torch.nn.Sequential(
    torch.nn.Linear(D, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D),
)

model_vel = torch.nn.Sequential(
    torch.nn.Linear(D, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D),
)

model_onOff = torch.nn.Sequential(
    torch.nn.Linear(D, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D),
)


loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate_note = LEARNING_RATE_NOTE
learning_rate_vel = LEARNING_RATE_VEL
learning_rate_onoff = LEARNING_RATE_ONOFF

loss_note_val = 0.
loss_vel_val = 0.
loss_onOff_val = 0.


for t in range(ITERATIONS):
    
    if LEARN_NOTES: 

        y_pred_note = model_note(x_note)
        loss_note = loss_fn(y_pred_note, y_note)

        loss_note_val =  loss_note.item()

        model_note.zero_grad()

        loss_note.backward()

        
        with torch.no_grad():
            for param in model_note.parameters():
                param -= learning_rate_note * param.grad


        if loss_note_val < THRESHOLD: 

            break



    if LEARN_VEL: 

        y_pred_vel = model_vel(x_vel)
        loss_vel = loss_fn(y_pred_vel, y_vel)

        loss_vel_val =  loss_vel.item()

        model_vel.zero_grad()

        loss_vel.backward()

        
        with torch.no_grad():
            for param in model_vel.parameters():
                param -= learning_rate_vel * param.grad

    if LEARN_ONOFF: 

        y_pred_onOff = model_onOff(x_onOff)

        loss_onOff = loss_fn(y_pred_onOff, y_onOff)

        loss_onOff_val =  loss_onOff.item()

        model_onOff.zero_grad()

        loss_onOff.backward()

        
        with torch.no_grad():
            for param in model_onOff.parameters():
                param -= learning_rate_onoff * param.grad


    print(t, loss_note_val, "   ", loss_vel_val, "    ", loss_onOff_val)

    if t % 100 == 0 : 
        learning_rate_note = alpha * learning_rate_note
        learning_rate_vel = alpha * learning_rate_vel
        learning_rate_onoff = alpha * learning_rate_onoff


############ END DEEP LEARNING ########################

#

if LEARN_NOTES: 

    y_pred_note = y_pred_note * 127
    y_pred_note_rect = toRectArray(y_pred_note.int())

if LEARN_VEL: 

    y_pred_vel = y_pred_vel * 127
    y_pred_vel_rect = toRectArray(y_pred_vel.int())


if LEARN_ONOFF: 
    y_pred_onOff = y_pred_onOff * (1./ 1000)
    y_pred_onOff_rect = toRectArray(y_pred_onOff.round().int())


    print("source > 0: ",  np.where( onOffArray_source > 0 )[0].shape)
    print("learned > 0: ", np.where( y_pred_onOff_rect > 0 )[0].shape)


# "_lossnote_" + str(round(loss_note_val, 2)) + "_lossvel_" + str(round(loss_vel_val, 2)) +  "_lossonOff_" + str(round(loss_onOff_val, 2)) +
filename = FILEPRE +  "_itr_" + str(ITERATIONS) + "_alpha_" + str(alpha) + "_learningrates_" + str(LEARNING_RATE_NOTE) + "_" + str(LEARNING_RATE_VEL) + "_" + str(LEARNING_RATE_ONOFF) + "_H_" + str(H) + "_" + FILEPOST + "_" + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

#midiFunctions.createMidi(noteArray_source, velocityArray_source, onOffArray_source, int(round(60000000 / TEMPO)), filename + "_nolearning" )

if LEARN_NOTES and not LEARN_VEL and not LEARN_ONOFF: 

    midiFunctions.createMidi(y_pred_note_rect, velocityArray_source, onOffArray_source, int(round(60000000 / TEMPO)), filename + "_learnnotesonly" )

    mid = mido.MidiFile("midi/" + filename + "_learnnotesonly" + '.mid')

    playMidi(mid)

elif LEARN_ONOFF and not LEARN_NOTES and not LEARN_VEL: 

    midiFunctions.createMidi(noteArray_source, velocityArray_source, y_pred_onOff_rect, int(round(60000000 / TEMPO)), filename + "_learnonoffonly"  )

    mid = mido.MidiFile("midi/" + filename + "_learnonoffonly" + '.mid')

    playMidi(mid)

elif LEARN_NOTES and LEARN_VEL and LEARN_ONOFF: 

    midiFunctions.createMidi(y_pred_note_rect, y_pred_vel_rect, y_pred_onOff_rect, int(round(60000000 / TEMPO)), filename + "_learnall3"  )

    mid = mido.MidiFile("midi/" + filename + "_learnall3" + '.mid')

    playMidi(mid)

elif LEARN_NOTES and not LEARN_VEL and LEARN_ONOFF :

    midiFunctions.createMidi(y_pred_note_rect, velocityArray_source, y_pred_onOff_rect, int(round(60000000 / TEMPO)), filename + "_nolearnvel"  )

    mid = mido.MidiFile("midi/" + filename + "_nolearnvel" + '.mid')

    playMidi(mid)

else: 

    print("didn't configure that option")

