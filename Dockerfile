FROM python:3.8

# instalación de librerías streamlit (solo en docker):
RUN pip install numpy streamlit

WORKDIR /app

# se copia desde el ámbito local al ámbito del contenedor (solo en docker):
COPY src/* /app/
COPY data/* /app/data/
COPY image/* /app/image/
COPY model/* /app/model/

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]