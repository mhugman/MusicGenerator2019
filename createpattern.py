

import random
import numpy as np

def createPattern(): 


    rhythm_pattern = np.random.choice(2, 4)

    note_pattern = np.random.choice(4, 4, replace=False)

    return rhythm_pattern, note_pattern



'''
rhythm_patterns = 

{
    
    0: [1,0,1,0], 
    1: [0,1,0,1], 
    2: [1,1,1,0], 
    3: [0,1,1,1], 
    4: [0,1,1,0], 
    5: [1,0,0,1], 
    6: [1,1,0,1], 
    7: [1,0,1,1]

}


note_patterns = 

{
    0: [0,1,2,3], 
    1: [0,2,1,3], 
    2: [0,3,1,2], 
    3: [1,2,3,0], 
    4: [1,3,0,2], 
    5: [3,0,1,2], 
    6: [3,2,0,1]

}
'''