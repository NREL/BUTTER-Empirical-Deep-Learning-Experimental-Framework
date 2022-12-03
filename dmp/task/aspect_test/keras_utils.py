import tensorflow.keras as keras


def count_vars_in_keras_model(model: keras.Model, var_getter) -> int:
    count = 0
    for var in var_getter(model):
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        count += acc
    return count


def count_trainable_parameters_in_keras_model(model: keras.Model) -> int:
    return count_vars_in_keras_model(model, lambda m: m.trainable_variables)


def count_parameters_in_keras_model(model: keras.Model) -> int:
    return count_vars_in_keras_model(model, lambda m: m.variables)


def count_non_trainable_parameters_in_keras_model(model: keras.Model) -> int:
    return count_vars_in_keras_model(model,
                                     lambda m: m.non_trainable_variables)
