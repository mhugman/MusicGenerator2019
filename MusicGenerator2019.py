import mido
import numpy as np
import torch
import math

import midiFunctions

SONG_LENGTH = 25000
NUM_TRACKS = 5
TEMPO = 900

# Mario about 900 BPM

def toSquareArray(rectArray): 

    D = int(math.ceil((SONG_LENGTH / NUM_TRACKS) / 2))

    squareArray = np.zeros((D, D)).astype("int")

    row = 0 
    col = 0 

    chunk = 1 

    running = True

    endOfSong = False
    endOfRow = False

    while(running):
        squareArray[row * NUM_TRACKS : row * NUM_TRACKS + NUM_TRACKS: , col * NUM_TRACKS : col * NUM_TRACKS + NUM_TRACKS]  = rectArray[:, chunk * NUM_TRACKS :  chunk * NUM_TRACKS + NUM_TRACKS]

            col += 1

            chunk += 1

        except: 

            row 



    

    


def playMidi(mid): 
    
    for message in mid.play():
        outport.send(message)



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

############ DEEP LEARNING ########################

H = 100


# Create random Tensors to hold inputs and outputs
x_note = torch.from_numpy(noteArray)
y_note = torch.randn(NUM_TRACKS, SONG_LENGTH)

x_vel = torch.from_numpy(velocityArray)
y_vel = torch.randn(NUM_TRACKS, SONG_LENGTH)

x_onoff = torch.from_numpy(onOffArray)
y_onoff = torch.randn(NUM_TRACKS, SONG_LENGTH)



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

learning_rate = 1e-4
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

print("noteArray_mario: ", noteArray_mario)

midiFunctions.createMidi(noteArray_mario, velocityArray_mario, onOffArray_mario, int(round(60000000 / TEMPO)), "new_mario")

#mid = mido.MidiFile('midi/new_song.mid')
mid = mido.MidiFile('midi/new_mario.mid')
#mid = MidiFile('mario.mid')

playMidi(mid)