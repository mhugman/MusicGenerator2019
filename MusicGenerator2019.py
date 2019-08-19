import mido
import numpy as np
import torch
import math
import sys

import midiFunctions

np.set_printoptions(threshold=sys.maxsize)

SONG_LENGTH = 120000
NUM_TRACKS = 3
TEMPO = 850

# Mario about 850 BPM

def toSquareArray(rectArray): 

    numChunks = int(math.ceil((SONG_LENGTH / NUM_TRACKS)))

    D = 0 

    while D ** 2 < SONG_LENGTH * NUM_TRACKS: 

        D += 1
    
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

    D = 0 

    while D ** 2 < SONG_LENGTH * NUM_TRACKS: 

        D += 1
    
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

noteArray_mario = np.zeros(noteArray.shape).astype("int")
velocityArray_mario = np.zeros(noteArray.shape).astype("int")
onOffArray_mario = np.zeros(noteArray.shape).astype("int")

#print("note Array shape: ", noteArray.shape)
#print("note Array mario shape: ", noteArray_mario.shape)

noteArray_mario, velocityArray_mario, onOffArray_mario = midiFunctions.parseMidi(noteArray_mario, velocityArray_mario, onOffArray_mario, mido.MidiFile('midi/mario.mid'))

noteArray_mario_square = toSquareArray(noteArray_mario)

#print("squareArray: ", noteArray_mario_square)

noteArray_mario_rect = toRectArray(noteArray_mario_square)

#print("noteArray_mario: ", noteArray_mario)
#print(noteArray_mario.shape)
#print("noteArray_mario_square: ", noteArray_mario_square)
#print(noteArray_mario_square.shape)
#print("noteArray_mario_rect: ", noteArray_mario_rect)
#print(noteArray_mario_rect.shape)

midiFunctions.createMidi(noteArray_mario_rect, velocityArray_mario, onOffArray_mario, int(round(60000000 / TEMPO)), "new_mario_rect")

#mid = mido.MidiFile('midi/new_song.mid')
mid = mido.MidiFile('midi/new_mario_rect.mid')
#mid = mido.MidiFile('midi/test2.mid')
#mid = mido.MidiFile('midi/mario.mid')

#playMidi(mid)



############ DEEP LEARNING ########################

# source: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

H = 100


# Create random Tensors to hold inputs and outputs
#x = torch.from_numpy(noteArray_square.astype("float")).float()
x = torch.randn(10, 10)
y = torch.from_numpy(noteArray_mario_square.astype("float")).float()

#x = x * (1./128)
y = y * (1./128)

D_in = x.size()[0]
D_out = y.size()[0]



# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

#learning_rate = 0.0000005
learning_rate = 0.0001
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

############ END DEEP LEARNING ########################

y_pred = y_pred * 127

y_pred_rect = toRectArray(y_pred.int())


midiFunctions.createMidi(y_pred_rect, velocityArray_mario, onOffArray_mario, int(round(60000000 / TEMPO)), "y_pred_rect")

mid = mido.MidiFile('midi/y_pred_rect.mid')

playMidi(mid)

