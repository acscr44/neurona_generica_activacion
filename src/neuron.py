"""
Módulo que contiene la clase Neuron.
"""

import numpy as np

# Objeto Neuron, cuyo constructor recibe un vector, un scalar y el nombre de una función de activación:
class Neuron:
    """
    Clase que representa una neurona.
    """

    # Diccionario para evitar los múltiples `if`:
    __funciones = {
        "sigmoid": lambda cls, z: cls.__sigmoid(z),
        "hiperbolic": lambda cls, z: cls.__hiperbolic(z),
        "relu": lambda cls, z: cls.__relu(z)
    }

    # Método de clase llamado sigmoid que recibe un escalar y devuelve el resultado tras aplicarle la función sigmoide:
    @classmethod
    def __sigmoid(cls, z):
        return 1 / (1 + np.exp(-z))

    # Método clase llamado hiperbolic que recibe un escalar y devuelve el resultado tras aplicarle la función hiperbólica:
    @classmethod
    def __hiperbolic(cls, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    # Método de clase llamado relu que recibe un escalar y devuelve el resultado tras aplicarle la función ReLU:
    @classmethod
    def __relu(cls, z):
        return np.maximum(0, z)





    def __init__(self, w, bias, func):
        self.__weights:np.ndarray = w
        self.__bias:float = bias
        self.__activation:str = func



        if func not in self.__funciones:
            raise ValueError(f"Función de activación {func} no implementada")

# Método llamado pred que recibe un vector de entrada y devuelve la salida de la neurona:
    def pred(self, input_data):
        """
        Devuelve la salida de la neurona.
        """
        z = np.dot(input_data, self.__weights) + self.__bias
        return self.__funciones[self.__activation](self.__class__, z)






# Método llamado get_params que devuelve los parámetros de la neurona:
    def get_params(self):
        """
        Devuelve los parámetros de la neurona.
        """
        return f"""* Parámetros del objeto {__class__.__name__}:\n /
        \tPesos={self.__weights}, \n\tBias={self.__bias}, \n /
        \tFunción de activation={self.__activation}"""


# Método llamado changePesos que recibiendo un np.array modifica el valor de self.w:
    def change_pesos(self, w):
        """
        Modifica el valor de self.w.
        """
        self.__weights = w

# Método llamado changeBias que un scalar y actualiza el parámetro b de la neurona:
    def change_bias(self, b):
        """
        Actualiza el parámetro b de la neurona.
        """
        self.__bias = b



# Método compatible con toString para ser usado en los métodos get_params():
    def __str__(self):
        return f"Neuron(w={self.__weights}, b={self.__bias}, activation={self.__activation})"
