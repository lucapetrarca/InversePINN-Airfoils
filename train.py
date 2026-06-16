import os
import sys
import pickle
import time
import torch
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath("."))
from src.model import InversePINN
from src.physics import calcular_loss

def entrenar_pinn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando Entrenamiento Inverse PINN ===")
    print(f"Dispositivo: {device}")
    
    #Se asignan las rutas de los archivos
    ruta_dataset = "datos/processed/ala_2d_processed_completo.csv"
    ruta_escala = "datos/processed/ala_2d_processed_completo_escala.pkl"

    #Se cargan los datos y la constante de normalización de la velocidad
    df = pd.read_csv(ruta_dataset)
    with open(ruta_escala, "rb") as f:
        dic_escala = pickle.load(f)
        U_ref = dic_escala['U_ref']

    print(f"Velocidad de referencia (U_ref) cargada: {U_ref:.2f} m/s")

    #Se establece una InversePINN con 5 capas y 128 neuronas
    pinn = InversePINN(num_capas=5, num_neuronas=128).to(device)
    
    #Se asigna el método ADAM como optimizador para la primera fase
    optimizer_adam = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    
    #Se asigna el método LBFGS como optimizador para el ajuste final
    optimizer_lbfgs = torch.optim.LBFGS(
        pinn.parameters(), 
        lr=1.0, 
        max_iter=200, 
        history_size=50, 
        line_search_fn="strong_wolfe"
    )
    
    epochs_adam = 10000
    start_time = time.time()

    #Fase 1: Se optimiza con ADAM con Muestreo Dinámico
    print(f"\nArrancando Fase 1: {epochs_adam} épocas con ADAM...")
    for epoch in range(epochs_adam + 1):
        if epoch < 3000:
            lambda_physics = 0.0
        elif epoch < 7000:
            lambda_physics = 0.01   #Se modifica el lambda_physics según la epoch
        else:
            lambda_physics = 0.1
            
        #Se realiza muestreo importante asignando puntos cercanos a las condiciones de borde
        zona_critica = df[df['implicit_distance'].abs() < 0.05]
        zona_lejana = df[df['implicit_distance'].abs() >= 0.05]
        batch_critico = zona_critica.sample(n=min(2500, len(zona_critica)))
        batch_lejano = zona_lejana.sample(n=min(2500, len(zona_lejana)))
        df_batch = pd.concat([batch_critico, batch_lejano]).sample(frac=1.0)

        x_t = torch.tensor(df_batch['x'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        y_t = torch.tensor(df_batch['y'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        ux_t = torch.tensor(df_batch['U_x'].values.reshape(-1, 1), dtype=torch.float32).to(device)
        uy_t = torch.tensor(df_batch['U_y'].values.reshape(-1, 1), dtype=torch.float32).to(device)

        optimizer_adam.zero_grad()
        ux_pred, uy_pred = pinn(x_t, y_t)
        loss_data = torch.mean((ux_pred - ux_t)**2) + torch.mean((uy_pred - uy_t)**2)
        
        if lambda_physics > 0.0:
            loss_physics = calcular_loss(pinn, x_t, y_t)
        else:
            loss_physics = torch.tensor(0.0, device=device)
            
        loss_total = loss_data + lambda_physics * loss_physics
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        optimizer_adam.step()
        
        if epoch % 1000 == 0:
            visc_real = pinn.obtener_viscosidad().item() * U_ref * 1.0
            print(f"ADAM  | Epoch {epoch:5d} | L_Total: {loss_total.item():.5f} | L_Data: {loss_data.item():.5f} | L_physics: {loss_physics.item():.6f} | ν_real: {visc_real:.6f}")

    #Fase 2: Se optimiza con LBFGS fijando muestra para consistencia geométrica
    print(f"\nFase LBFGS Refinando curvatura de segundo orden...")
    pinn.train()
    
    #Se realiza muestreo estratificado para LBFGS asignando la mitad de la muestra a la capa limite
    zona_critica_lbfgs = df[df['implicit_distance'].abs() < 0.05]
    zona_lejana_lbfgs = df[df['implicit_distance'].abs() >= 0.05]
    batch_lbfgs_critico = zona_critica_lbfgs.sample(n=min(7500, len(zona_critica_lbfgs)))
    batch_lbfgs_lejano = zona_lejana_lbfgs.sample(n=min(7500, len(zona_lejana_lbfgs)))
    df_lbfgs = pd.concat([batch_lbfgs_critico, batch_lbfgs_lejano]).sample(frac=1.0)

    x_l = torch.tensor(df_lbfgs['x'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
    y_l = torch.tensor(df_lbfgs['y'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
    ux_l = torch.tensor(df_lbfgs['U_x'].values.reshape(-1, 1), dtype=torch.float32).to(device)
    uy_l = torch.tensor(df_lbfgs['U_y'].values.reshape(-1, 1), dtype=torch.float32).to(device)

    def closure():
        optimizer_lbfgs.zero_grad()
        ux_p, uy_p = pinn(x_l, y_l)
        l_data = torch.mean((ux_p - ux_l)**2) + torch.mean((uy_p - uy_l)**2)
        l_phys = calcular_loss(pinn, x_l, y_l)
        l_total = l_data + 0.1 * l_phys
        l_total.backward()
        return l_total

    optimizer_lbfgs.step(closure)
    
    ux_final, uy_final = pinn(x_l, y_l)
    l_data_final = torch.mean((ux_final - ux_l)**2) + torch.mean((uy_final - uy_l)**2)
    l_phys_final = calcular_loss(pinn, x_l, y_l)
    l_total_final = l_data_final + 0.1 * l_phys_final
    visc_final = pinn.obtener_viscosidad().item() * U_ref * 1.0
    
    print(f"\nLBFGS | Iteración Final | L_Total: {l_total_final.item():.5f} | L_Data: {l_data_final.item():.5f} | L_physics: {l_phys_final.item():.6f} | ν_real definitiva: {visc_final:.6f}")

    print("\n¡Entrenamiento híbrido finalizado!")
    
    #Se guardan los pesos
    ruta_guardado = "datos/processed/inverse_pinn_model.pth"
    torch.save(pinn.state_dict(), ruta_guardado)
    print(f"Pesos del modelo guardados exitosamente en: {ruta_guardado}")

if __name__ == "__main__":
    entrenar_pinn()