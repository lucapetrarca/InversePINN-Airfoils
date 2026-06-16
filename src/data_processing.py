import pandas as pd
import pickle

def procesar_y_guardar_datos(ruta_raw, ruta_processed, ruta_toy):
    df = pd.read_csv(ruta_raw)
    cols = ['x', 'y', 'U_x', 'U_y', 'nut', 'implicit_distance']
    df = df[cols].copy()

    tol = 1e-4
    df['es_borde'] = df['implicit_distance'].abs() < tol 

    #Se escala el campo vectorial con la velocidad máxima absoluta
    U_ref = max(df['U_x'].abs().max(), df['U_y'].abs().max())

    df['U_x'] = df['U_x'] / U_ref
    df['U_y'] = df['U_y'] / U_ref
    #No se escalan x e y, manteniendo la geometría del ala

    df.to_csv(ruta_processed, index=False)
    
    with open(ruta_processed.replace('.csv', '_escala.pkl'), 'wb') as f:
        pickle.dump({'U_ref': U_ref}, f)

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
    procesar_y_guardar_datos(raw, processed, toy)