import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pickle

def procesar_y_guardar_datos(ruta_raw, ruta_processed, ruta_toy):
    df = pd.read_csv(ruta_raw)

    cols = ['x', 'y', 'U_x', 'U_y', 'nut', 'implicit_distance']
    df = df[cols].copy()

    tol = 1e-4
    df['es_borde'] = df['implicit_distance'].abs() < tol 

    scaler = MinMaxScaler(feature_range=(-1, 1))
    cols_a_escalar = ['x', 'y', 'U_x', 'U_y']
    df[cols_a_escalar] = scaler.fit_transform(df[cols_a_escalar])

    df.to_csv(ruta_processed, index=False)
    
    with open(ruta_processed.replace('.csv', '_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    df_borde_toy = df[df['es_borde'] == True].sample(n=250, random_state=42)
    df_interior_toy = df[df['es_borde'] == False].sample(n=250, random_state=42)

    df_toy = pd.concat([df_borde_toy, df_interior_toy], ignore_index=True)

    df_toy = df_toy.sample(frac=1, random_state=42).reset_index(drop=True)

    df_toy.to_csv(ruta_toy, index=False)

    return df

if __name__ == "__main__":
    raw = "datos/raw/ala_2d_completo.csv"
    processed = "datos/processed/ala_2d_processed_completo.csv"
    toy = "datos/toy/ala_2d_toy.csv"
    
    print("Procesando datos y guardando scaler...")
    procesar_y_guardar_datos(raw, processed, toy)
    print("¡Todo guardado con éxito!")