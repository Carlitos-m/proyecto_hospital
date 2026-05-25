from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd

app = FastAPI()

# Configura tu cadena de conexión (reemplaza con tus datos de MySQL)
# El formato es: mysql+mysqlconnector://usuario:contraseña@servidor/nombre_de_tu_db
DATABASE_URL = "mysql+mysqlconnector://root:tu_contraseña@localhost/hosp_project"
engine = create_engine(DATABASE_URL)

@app.get("/")
def read_root():
    return {"mensaje": "¡Conexión exitosa! Prueba ir a /datos-transacciones"}
@app.get("/datos-transacciones")
def obtener_transacciones():
    # Gracias a que ya limpiamos las fechas, podemos hacer queries súper rápidos
    query = """
    SELECT fecha_real, fecha_hora_real 
    FROM hostransacciones 
    WHERE fecha_real IS NOT NULL 
    LIMIT 5
    """
    df = pd.read_sql(query, engine)

    # FastAPI convierte automáticamente el DataFrame a formato JSON para tu dashboard
    return df.to_dict(orient="records")