import numpy as np

# Objeto Neuron, cuyo constructor recibe un vector, un scalar y el nombre de una función de activación:
class Neuron:

    # Diccionario para evitar los múltiples `if`:
    __funciones = {
        "sigmoid": lambda z: Neuron.__sigmoid(z),
        "hiperbolic": lambda z: Neuron.__hiperbolic(z),
        "relu": lambda z: Neuron.__relu(z)
    }

    # Método de clase llamado sigmoid que recibe un escalar y devuelve el resultado tras aplicarle la función sigmoide:
    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Método clase llamado hiperbolic que recibe un escalar y devuelve el resultado tras aplicarle la función hiperbólica:
    @staticmethod
    def __hiperbolic(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    # Método de clase llamado relu que recibe un escalar y devuelve el resultado tras aplicarle la función ReLU:
    @staticmethod
    def __relu(z):
        return np.maximum(0, z)





    def __init__(self, w, bias, func):
        self.__weights:np.ndarray = w
        self.__bias:float = bias
        self.__activation:str = func

        # self.__funciones = {
        #     "sigmoid": self.__sigmoid,
        #     "hiperbolic": self.__hiperbolic,
        #     "relu": self.__relu
        # }

        if func not in self.__funciones:
            raise ValueError(f"Función de activación {func} no implementada")

# Método llamado pred que recibe un vector de entrada y devuelve la salida de la neurona:
    def pred(self, input_data):
        z = np.dot(input_data, self.__weights) + self.__bias
        return self.__funciones[self.__activation](z)


    
# Método llamado get_params que devuelve los parámetros de la neurona:
    def get_params(self):
        return f"* Parámetros del objeto {self.__class__.__name__}:\n\tPesos={self.__weights}, \n\tBias={self.__bias}, \n\tFunción de activation={self.__activation}"


# Método llamado changePesos que recibiendo un np.array modifica el valor de self.w:
    def changePesos(self, w):
        self.__weights = w

# Método llamado changeBias que un scalar y actualiza el parámetro b de la neurona:
    def changeBias(self, b):
        self.__bias = b



# Método compatible con toString para ser usado en los métodos get_params():
    def __str__(self):
        return f"Neuron(w={self.__weights}, b={self.__bias}, activation={self.__activation})"