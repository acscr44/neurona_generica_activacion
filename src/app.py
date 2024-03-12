"""
Docstring app.py
"""
import streamlit as st
import numpy as np
from neuronstatic import Neuron

# Estilos  ###################################################################################


style_width = """
    <style>
        .appview-container  .main  .block-container{
            max-width: 60%;
        }
    </style>
    """
style_text = """
    <style>
    .normal-font {
        font-size:14px;
        font-family: "Source Code Pro", monospace;
    }
    </style>
    """
# st.set_page_config(layout="wide")

st.markdown(style_width, unsafe_allow_html=True)

# Cabecera  ##################################################################################

st.image("image/neurona.jpg")
st.title('Simulador de neuronas')

# Selector de pesos/entradas  ################################################################
input_number:int = st.slider(label='Elige el número de entradas/pesos que tendrá la neurona', 
                             label_visibility='visible', 
                             min_value=1, max_value=10)

# Variables inicializadas para pesos y entradas:
var_peso = ['0.0'] * input_number
var_entrada = ['0.0'] * input_number


# Estructura de columnas para Pesos
st.subheader('Pesos')
col_pesos = st.columns(input_number)

# Creación de inputs dentro de cada columna
for c in range(input_number):
    with col_pesos[c]:
        st.markdown(f'w<sub>{c + 1}</sub>', unsafe_allow_html=True)
        # Los tipo float generan a veces números con muchos decimales:
        peso = st.number_input(f'w {c + 1}', key=f'peso_w{c}', value=0.0, label_visibility='collapsed')
        var_peso[c] =  round(peso, 2)   # Solución: redondeo a dos decimales

# st.write("w = ", str(var_peso))
html_content_weight = f"""
<div class='normal-font'>
    w = {str(var_peso)}
</div>
""".replace("\n", '')
st.markdown(style_text + html_content_weight, unsafe_allow_html=True)

# Estructura de columnas para Entradas
st.subheader('Entradas')
col_entradas = st.columns(input_number)

# Creación de inputs dentro de cada columna
for c in range(input_number):
    with col_entradas[c]:
        st.markdown(f'x<sub>{c + 1}</sub>', unsafe_allow_html=True)
        # Los tipo float generan a veces números con muchos decimales:
        entrada = st.number_input(f'Entrada {c + 1}', key=f'entrada_x{c}', value=0.0, label_visibility='collapsed')
        var_entrada[c] =  round(entrada, 2)   # Solución: redondeo a dos decimales

# st.write("w = ", str(var_peso))
html_content_input = f"""
<div class='normal-font'>
    x = {str(var_entrada)}
</div>
""".replace("\n", '')
st.markdown(style_text + html_content_input, unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.subheader('Sesgo')
    bias = st.number_input("Introduzca el valor del sesgo", key='bias')

with col2:
    st.subheader('Función de Activación')
    ml_model = ["Sigmoide", "ReLU", "Tangente hiperbólica"]
    opcion = st.selectbox('Elige la función de activación', ml_model, key="option")

# Cálculo de la salida
# var_peso = ['0.25', '0.65']
# var_entrada = ['3', '5']
# var_peso = np.array(var_peso, dtype=float)
# var_entrada = np.array(var_entrada, dtype=float)

if st.button("Calcular la salida", key='submit'):
    ml_func_activ = ["sigmoid", "relu", "hiperbolic"]
    func_activ = ml_func_activ[ml_model.index(opcion)]
    p1 = Neuron(var_peso, bias, func_activ)
    salida = p1.pred(var_entrada)
    st.text(f"La salida de la neurona es {salida}")
