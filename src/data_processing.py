import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def procesar_y_guardar_datos(ruta_raw, ruta_processed, ruta_toy):
    df = pd.read_csv(ruta_raw)

    cols = ['x', 'y', 'U_x', 'U_y', 'nut', 'implicit_distance']
    df = df[cols].copy()

    #Separamos las condiciones de borde, implicit distance cercano a 0
    tol = 1e-4
    #Creamos columna booleana para saber qué puntos pertenecen al ala
    df['es_borde'] = df['implicit_distance'].abs() < tol 

    #Escalamos a [-1, 1] todo el dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    cols_a_escalar = ['x', 'y', 'U_x', 'U_y']
    df[cols_a_escalar] = scaler.fit_transform(df[cols_a_escalar])

    df.to_csv(ruta_processed, index=False)

    #Para probar con datos toy, asignamos 250 muestras del ala y 250 del interior
    df_borde_toy = df[df['es_borde'] == True].sample(n=250, random_state=42)
    df_interior_toy = df[df['es_borde'] == False].sample(n=250, random_state=42)

    #Los unimos
    df_toy = pd.concat([df_borde_toy, df_interior_toy], ignore_index=True)

    #Mezclamos los datos toy
    df_toy = df_toy.sample(frac=1, random_state=42).reset_index(drop=True)

    #Lo guardamos
    df_toy.to_csv(ruta_toy, index=False)

    return df