import os
import sys
import pickle
import time
import torch
import pandas as pd
import numpy as np

# Asegurar que el script encuentre el paquete src
sys.path.append(os.path.abspath("."))
from src.model import InversePINN
from src.physics import calcular_loss

def test_entrenamiento_rapido():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando Test Rápido ===")
    print(f"Dispositivo: {device}")
    
    # 1. Rutas de archivos (ajustá si es necesario)
    ruta_dataset = "datos/processed/ala_2d_processed_completo.csv" # O podés usar el processed completo
    ruta_scaler = "datos/processed/ala_2d_processed_completo_scaler.pkl"

    # 2. Cargar datos y scaler
    df = pd.read_csv(ruta_dataset)
    with open(ruta_scaler, "rb") as f:
        scaler = pickle.load(f)

    # 3. Extraer factores del scaler
    escalas = (scaler.scale_[0], scaler.scale_[1], scaler.scale_[2], scaler.scale_[3])
    minimos_u = (scaler.min_[2], scaler.min_[3])

    # 4. Instanciar Modelo y Optimizador
    pinn = InversePINN(num_capas=4, num_neuronas=64).to(device)
    
    # lr=3e-4 ajustado para mejor convergencia y estabilidad
    optimizer = torch.optim.Adam(pinn.parameters(), lr=3e-4) 
    
    epochs = 5000
    #lambda_physics = 1e-3

    print(f"\nArrancando bucle de {epochs} épocas con muestreo dinámico...")
    start_time = time.time()

    # 5. Bucle de Entrenamiento
    for epoch in range(epochs + 1):
                # Adentro del for epoch in range(epochs):
        if epoch < 2000:
            lambda_physics = 1e-3
        elif epoch < 4000:
            lambda_physics = 1e-2
        else:
            lambda_physics = 1e-1  # Le damos mucha más fuerza a Burgers al final
        # --- MUESTREO DINÁMICO ---
        # En cada época, elegimos 5000 puntos distintos al azar
        df_batch = df.sample(n=min(5000, len(df)))
        
        # Transformar a tensores
        x_t = torch.tensor(df_batch['x'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        y_t = torch.tensor(df_batch['y'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        ux_t = torch.tensor(df_batch['U_x'].values.reshape(-1, 1), dtype=torch.float32).to(device)
        uy_t = torch.tensor(df_batch['U_y'].values.reshape(-1, 1), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        
        # Forward pass
        ux_pred, uy_pred = pinn(x_t, y_t)
        
        # Calcular pérdidas
        loss_data = torch.mean((ux_pred - ux_t)**2) + torch.mean((uy_pred - uy_t)**2)
        loss_physics = calcular_loss(pinn, x_t, y_t, escalas=escalas, minimos_u=minimos_u)
        
        loss_total = loss_data + lambda_physics * loss_physics
        
        # Backpropagation
        loss_total.backward()
        
        # --- GRADIENT CLIPPING ---
        # Se aplica SIEMPRE después del .backward() y antes del .step()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Imprimir progreso cada 500 épocas
        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            visc_actual = pinn.obtener_viscosidad().item()
            print(f"Epoch {epoch:4d} | Tiempo: {elapsed:.1f}s | Loss Total: {loss_total.item():.6f} | L_Data: {loss_data.item():.6f} | L_Phys: {loss_physics.item():.2f} | ν: {visc_actual:.6f}")

    print("\n¡Test finalizado con éxito!")

if __name__ == "__main__":
    test_entrenamiento_rapido()