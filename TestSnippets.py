__author__ = 'Thush'

import theano
import numpy as np
import theano.tensor as T

def testTensorSwitch():

    x = T.dmatrix('x')

    func = theano.function(
        inputs=[x],
        outputs=[T.switch(x<0,0,x)]
    )

    arr = func(np.asarray([[0.5,0.4,-0.1],[-0.2,0.2,4],[1,2,3]],dtype=theano.config.floatX))
    print(arr)


if __name__=='__main__':
    testTensorSwitch()