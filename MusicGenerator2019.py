import mido
import numpy as np
import torch
import math
import sys
import datetime

import midiFunctions

np.set_printoptions(threshold=sys.maxsize)


############ GLOBAL PARAMETERS ######################

SONG_LENGTH = 120000 # About 120000 for mario
NUM_TRACKS = 4
TEMPO = 850 # mario about 850 BPM

H = 100 # dimensions for hidden layer
ITERATIONS = 1000
LEARNING_RATE = 0.0001
FILEPRE = "emulate_mario"
FILEPOST = "all3"

FILESOURCE = "mario"

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

            squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS] = rectArray[:, NUM_TRACKS * chunk : NUM_TRACKS * chunk + NUM_TRACKS]
            
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


            rectArray[:, NUM_TRACKS * chunk : NUM_TRACKS * chunk + NUM_TRACKS] = squareArray[NUM_TRACKS * i : NUM_TRACKS * i + NUM_TRACKS ,NUM_TRACKS * j : NUM_TRACKS * j + NUM_TRACKS]

            #print(rectArray[:,:NUM_TRACKS * chunk + NUM_TRACKS])

            chunk += 1

    return rectArray



def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)



outport = mido.open_output()

noteArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
velocityArray = np.random.randint(0, 128, size=(NUM_TRACKS, SONG_LENGTH))
onOffArray = np.random.randint(-1, 2, size=(NUM_TRACKS, SONG_LENGTH))

noteArray_square = toSquareArray(noteArray)

#print("noteArray: ", noteArray)


midiFunctions.createMidi(noteArray, velocityArray, onOffArray, int(round(60000000 / TEMPO)), "new_song")

noteArray_source = np.zeros(noteArray.shape).astype("int")
velocityArray_source = np.zeros(noteArray.shape).astype("int")
onOffArray_source = np.zeros(noteArray.shape).astype("int")

#print("note Array shape: ", noteArray.shape)
#print("note Array source shape: ", noteArray_source.shape)

noteArray_source, velocityArray_source, onOffArray_source = midiFunctions.parseMidi(noteArray_source, velocityArray_source, onOffArray_source, mido.MidiFile('midi/' + FILESOURCE + '.mid'))

noteArray_source_square = toSquareArray(noteArray_source)

velocityArray_source_square = toSquareArray(velocityArray_source)

onOffArray_source_square = toSquareArray(onOffArray_source)

#print("squareArray: ", noteArray_source_square)

#noteArray_source_rect = toRectArray(noteArray_source_square)

#print("noteArray_source: ", noteArray_source)
#print(noteArray_source.shape)
#print("noteArray_source_square: ", noteArray_source_square)
#print(noteArray_source_square.shape)
#print("noteArray_source_rect: ", noteArray_source_rect)
#print(noteArray_source_rect.shape)

#midiFunctions.createMidi(noteArray_source_rect, velocityArray_source, onOffArray_source, int(round(60000000 / TEMPO)), "new_source_rect")

#mid = mido.MidiFile('midi/new_song.mid')
#mid = mido.MidiFile('midi/new_source_rect.mid')
#mid = mido.MidiFile('midi/test2.mid')
#mid = mido.MidiFile('midi/source.mid')

#playMidi(mid)



############ DEEP LEARNING ########################

# source: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

#print("onOffArray_source:", onOffArray_source)


# Create random Tensors to hold inputs and outputs
#x = torch.from_numpy(noteArray_square.astype("float")).float()

x_note = torch.randn(D, D)
x_vel = torch.randn(D, D)
x_onOff = torch.zeros(D, D) + 0.5

#print("velocityArray_source: ", velocityArray_source)

#print("velocityArray_source_square: ", velocityArray_source_square)

y_note = torch.from_numpy(noteArray_source_square.astype("float")).float()
y_vel = torch.from_numpy(velocityArray_source_square.astype("float")).float()
y_onOff = torch.from_numpy(onOffArray_source_square.astype("float")).float()

#print("y_vel: ", y_vel)

#x = x * (1./128)
y_note = y_note * (1./128)
y_vel = y_vel * (1./128)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
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

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')


for t in range(ITERATIONS):

    y_pred_note = model_note(x_note)
    y_pred_vel = model_vel(x_vel)
    y_pred_onOff = model_onOff(x_onOff)

    #print(y_pred_onOff)

    loss_note = loss_fn(y_pred_note, y_note)
    loss_vel = loss_fn(y_pred_vel, y_vel)
    loss_onOff = loss_fn(y_pred_onOff, y_onOff)

    print(t, loss_note.item(), loss_vel.item(), loss_onOff.item())

    model_note.zero_grad()

    loss_note.backward()

    
    with torch.no_grad():
        for param in model_note.parameters():
            param -= LEARNING_RATE * param.grad

    model_vel.zero_grad()

    loss_vel.backward()

    
    with torch.no_grad():
        for param in model_vel.parameters():
            param -= LEARNING_RATE * param.grad

    model_onOff.zero_grad()

    loss_onOff.backward()

    
    with torch.no_grad():
        for param in model_onOff.parameters():
            param -= LEARNING_RATE * param.grad

############ END DEEP LEARNING ########################

y_pred_note = y_pred_note * 127
y_pred_vel = y_pred_vel * 127
#y_pred_onOff = y_pred_onOff + 0.6

#print(y_pred_onOff)

#print("y_pred_onOff rounded int: ", y_pred_onOff.round().int())


y_pred_note_rect = toRectArray(y_pred_note.int())
y_pred_vel_rect = toRectArray(y_pred_vel.int())
y_pred_onOff_rect = toRectArray(y_pred_onOff.round().int())

print(np.where( y_pred_onOff_rect > 0 ))

#print("y_pred_note_rect: ", y_pred_note_rect)
#print("y_pred_vel_rect: ", y_pred_vel_rect)
#print("y_pred_onOff_rect: ", y_pred_onOff_rect)

filename = FILEPRE + "_" + str(ITERATIONS) + "_" + str(LEARNING_RATE) + "_" + FILEPOST + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


midiFunctions.createMidi(y_pred_note_rect, y_pred_vel_rect, y_pred_onOff_rect, int(round(60000000 / TEMPO)), filename )

mid = mido.MidiFile('midi/' + filename + '.mid')

playMidi(mid)

