""" Models implementation.
"""
from keras import backend as K


# TODO: rewrite to factory
def build_model(params, in_dim=None, learning_config=None, impl='torch'):
    """The interface that should be used to obtain the instance of
        Model class with provided parameters.
    
    Args:
        params: A tuple with neural network paramters 
            with the following format: (
                <number of convolutional layers
                    if any, 0 otherwise>,
                <number of filters in each convolutional layer
                    if any, 0 otherwise>,
                <number of fully-connected layers>,
                <number of neurons in each fully-connected layer>
            ).

    Returns:
        An instance of Model class that represents
            a model with corresponding parameters.
    """
    K.clear_session()

    if len(params) == 2:
        params = (0, 0, *params)

    if impl == 'torch':
        from ml.model_torch import Model
        print('DEBUG: torch implementation')
        model = Model(*params, in_dim, learning_config)
    else:
        from ml.model_keras import Model
        print('DEBUG: keras implementation')
        model = Model(*params, learning_config)

    return model
