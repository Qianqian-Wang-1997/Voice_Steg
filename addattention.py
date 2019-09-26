from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers


class AttLayer(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('normal')#keras1.2.2
        self.init = initializers.get('normal')

        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 1)))
        # self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        # be sure you call this somewhere!
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # eij = K.tanh(K.dot(x, self.W))
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        # weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weights = ai / K.expand_dims(K.sum(ai, axis=1), 1)
        # weighted_input = x * weights.dimshuffle(0, 1, 'x')
        weighted_input = x * weights

        # return weighted_input.sum(axis=1)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
