import tensorflow as tf

class DLT(tf.keras.layers.Layer):
    def __init__(self, num_iters = 2, **kwargs):
        super(DLT, self).__init__(**kwargs)
        self.num_iters = num_iters

    def get_config(self):
        config = super(DLT, self).get_config()
        config.update({"num_iters": self.num_iters})
        return config

    def homogeneous_to_euclidean(self,v):
        if len(v.shape) > 1:
            return tf.transpose(tf.transpose(v[:,:-1])/v[:,-1])
        else:
            return v[:-1]/v[-1]

    def call(self, input):

        """
        This call accepts batched Projection matrices and batched uv coordinates from multiple cameras and return batched 3D coordinates.
        Any number of views >= 2 is supported.
        Projection matrices M = input[0]
        uv coordinate pts = input[1]

        Input shape:
        M = [batch_size, camera_view_index, 3, 4]
        pts = [batch_size, camera_view_index, 2]
        If there are 4 camera views, then M.shape = [batch_size, 4, 3, 4] and pts.shape = [batch_size, 4, 2]
        """

        M = input[0]
        pts = input[1]

        batch_size = tf.shape(M)[0] #batch size can change dynamically
        views = M.shape[1]

        #We need to solve Ax = 0. In this case, A = (u x P) for a single camera, with u = camera pixel coords, P = camera projection matrix
        #However, only A[0:2] is needed, since the last row is a linear combination of the first 2.
        A = tf.repeat(M[:,:,2:3], repeats = 2, axis = 2) * tf.reshape(pts, (batch_size, pts.shape[1], 2, 1)) - M[:,:,0:2]
        A = tf.reshape(A, (batch_size, 2 * views, 4))

        #contruct (A - alpha * I) for SII
        alpha = tf.constant(0.001)
        B = tf.transpose(A, perm = [0,2,1]) @ A - tf.repeat(tf.reshape(alpha * tf.eye(4), (1,4,4)), repeats = batch_size, axis = 0)

        #initial guess for triangulated point is randomly generated.
        #A good starting point is near [0,0,0,1], assuming the world coords are setup to be (0,0,0) at the center of the real camera space
        X = tf.random.normal((batch_size, 4, 1), mean = 0.5, stddev = 0.5 , dtype = 'float32')

        #solve By = X for y. From SII, X = y/|y|
        for _ in range(self.num_iters):
            X =  tf.linalg.solve(B, X)
            X = X/tf.expand_dims(tf.norm(X, axis = 1), axis = -1)

        X = tf.reshape(X, (batch_size, 4))

        return self.homogeneous_to_euclidean(X)
