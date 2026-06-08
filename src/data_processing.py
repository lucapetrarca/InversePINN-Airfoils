import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def procesar_y_guardar_datos(ruta_raw, ruta_processed, ruta_toy, num_muestras=5000):
    #Cargamos datos crudos
    df = pd.read_csv(ruta_raw)

    #Nos quedamos con las columnas que nos importan: x, y espaciales; U_x, U_y velocidades; nut viscosidad; implicit_distance al ala
    cols = ['x', 'y', 'U_x', 'U_y', 'nut', 'implicit_distance']
    df = df[cols].copy()

    #Nuestra condición de borde es la superficie del ala
    #Elegimos los puntos que están casi pegados a la chapa (tolerancia cercana a 0) como el borde, y el resto en el interior del fluido
    tol = 1e-4
    df_borde = df[df['implicit_distance'].abs() < tol].copy()
    df_interior = df[df['implicit_distance'].abs() >= tol].copy()

    #Para no calcular demasiadas derivadas, submuestreamos el interior (tenía 140000 puntos) y el borde (el borde tenía 20000 puntos)
    num_interior = num_muestras
    num_borde = num_muestras
    df_borde = df_borde.sample(n=min(len(df_borde), num_borde), random_state=42)
    df_interior = df_interior.sample(n=min(len(df_interior), num_interior), random_state=42) #Ver de nuevo el método de muestreo

    #Nos quedamos con la muestra de 5000 puntos y con las condiciones de borde
    df_final = pd.concat([df_borde, df_interior], ignore_index=True)

    #Normalizamos a [-1, 1] usando MinMax Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    cols_a_escalar = ['x', 'y', 'U_x', 'U_y']
    df_final[cols_a_escalar] = scaler.fit_transform(df_final[cols_a_escalar])

    #Guardamos el dataset procesado
    df_final.to_csv(ruta_processed, index=False)

    #Hacemos y guardamos un dataset de "juguete" para realizar pruebas->Son 100 condiciones de borde (las que entran primero)
    df_toy = df_final.head(100)
    df_toy.to_csv(ruta_toy, index=False)

    return df_final