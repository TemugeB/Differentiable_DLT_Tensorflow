import numpy as np
import tensorflow as tf
from diffDLT import DLT

def _get_points_for_demo(batch_size = 10):

    #camera 1 rotation and translation matrix.
    RT_c1 = tf.constant([[1 , 0 , 0 , 0 ],
                         [0 , 1 , 0 , 0 ],
                         [0 , 0 , 1 , 10]], dtype = 'float32')

    #camera matrix. Assume same camera
    CM = tf.constant([[50, 0 , 128],
                      [0 , 50, 128],
                      [0 , 0 , 1 ]], dtype = 'float32')

    #Camera 2 rot and trans
    RT_c2 = tf.constant([[1 , 0 , 0 , 0 ],
                         [0 , 0 , 1 , 0 ],
                         [0, -1 , 0 , 10]], dtype = 'float32')

    #projection matrices
    P1 = CM @ RT_c1
    P2 = CM @ RT_c2

    #generate some points
    Xs = 5 * tf.random.uniform((batch_size, 3))
    #add homogeneous coordinate
    Xs = tf.concat([Xs, tf.ones((batch_size, 1))], axis = -1)
    #print(Xs)

    #get camera pixel coords
    uvs1 = P1 @ tf.reshape(Xs, (batch_size, 4, 1))
    uvs2 = P2 @ tf.reshape(Xs, (batch_size, 4, 1))

    #remove homogeneous coordinate
    uvs1 = tf.reshape(uvs1[:,:-1], (batch_size, 2))/uvs1[:,-1]
    uvs2 = tf.reshape(uvs2[:,:-1], (batch_size, 2))/uvs2[:,-1]

    return Xs, P1, P2, uvs1, uvs2

def _triangulation_demo():

    Xs, P1, P2, uvs1, uvs2 = _get_points_for_demo(10)

    dlt = DLT()

    #prepare pts to pass to DLT
    uvs1 = tf.reshape(uvs1, (-1, 1, 2))
    uvs2 = tf.reshape(uvs2, (-1, 1, 2))
    pts = tf.concat([uvs1, uvs2], axis = 1) #[batch_size, cam_views, 2]
    #prepare projection matrices
    M = tf.stack([P1, P2], axis = 0)
    M = tf.repeat(tf.expand_dims(M, axis = 0), axis = 0, repeats = uvs1.shape[0])

    #create TF.keras layer.
    dlt = DLT()

    triangulations = dlt([M, pts])

    print('\n Triangulation testing: ', 20 * '-')

    for _x, _tri in zip(Xs, triangulations):
        print('x:', _x[:3].numpy(), ', triangulated:', _tri.numpy(), ', MSE error:', tf.reduce_mean(tf.square(_x[:3] - _tri)).numpy())

def _gradient_descent_demo():

    Xs, P1, P2, uvs1, uvs2 = _get_points_for_demo(10)

    #add some random shift to uvs. Idea is to gradient descent back to original uvs and to correct 3D points.
    uvs1_error = uvs1 + tf.random.normal((uvs1.shape), stddev = 2, dtype = 'float32')
    uvs2_error = uvs2 + tf.random.normal((uvs2.shape), stddev = 2, dtype = 'float32')

    #prepare pts to pass to DLT
    uvs1_error = tf.reshape(uvs1_error, (-1, 1, 2))
    uvs2_error = tf.reshape(uvs2_error, (-1, 1, 2))
    pts = tf.concat([uvs1_error, uvs2_error], axis = 1) #[batch_size, cam_views, 2]
    #prepare projection matrices
    M = tf.stack([P1, P2], axis = 0)
    M = tf.repeat(tf.expand_dims(M, axis = 0), axis = 0, repeats = uvs1.shape[0])

    #convert pts to differentiable tensors
    pts = tf.Variable(pts)

    epochs = 50
    optimizer = tf.keras.optimizers.Adam(lr = 0.01)

    dlt = DLT()

    # print('input camera pixel points:', pts.shape)
    # print('corresponding projection matrices:', M.shape)

    for ep in range(epochs):

        with tf.GradientTape() as tape:
            triangulations = dlt([M, pts]) # this line can be added to your model definition
            MSE = tf.reduce_mean(tf.square(Xs[:,:3] - triangulations)) #triangulations are not in homogeneous coordinates. Remove homogeneous coord from Xs.

        grads = tape.gradient(MSE, [pts])

        print('loss: ', MSE.numpy())
        optimizer.apply_gradients(zip(grads, [pts]))

    pass


if __name__ == '__main__':

    '''
    This file contains a triangulation demo and gradient descents demo. Uncomment which one you want to see.
    '''

    #_triangulation_demo()
    _gradient_descent_demo()
