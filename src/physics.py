import torch

def calcular_loss(modelo, x, y, escalas=(1.0, 1.0, 1.0, 1.0), minimos_u=(0.0, 0.0)):
    """    
    Args:
        modelo: Instancia de InversePINN.
        x: Tensor de coordenadas X (N, 1) con requires_grad=True.
        y: Tensor de coordenadas Y (N, 1) con requires_grad=True.
        escalas: Tupla con los factores (S_x, S_y, S_ux, S_uy) del MinMaxScaler.
        minimos_u: Tupla con los valores (min_ux, min_uy) del MinMaxScaler.
    """
    
    ux, uy = modelo(x, y)
    nu = modelo.obtener_viscosidad()
    
    # Extraemos las constantes del scaler
    sx, sy, sux, suy = escalas
    min_ux, min_uy = minimos_u
  
    ones = torch.ones_like(ux)
    
    # --- Primeras derivadas (en dominio escalado) ---
    dux_g = torch.autograd.grad(ux, [x, y], grad_outputs=ones, create_graph=True, allow_unused=True)
    dux_dx = dux_g[0] if dux_g[0] is not None else torch.zeros_like(ux)
    dux_dy = dux_g[1] if dux_g[1] is not None else torch.zeros_like(ux)
    
    duy_g = torch.autograd.grad(uy, [x, y], grad_outputs=ones, create_graph=True, allow_unused=True)
    duy_dx = duy_g[0] if duy_g[0] is not None else torch.zeros_like(ux)
    duy_dy = duy_g[1] if duy_g[1] is not None else torch.zeros_like(ux)
    
    # --- Laplacianos (en dominio escalado) ---
    dux_dxx_g = torch.autograd.grad(dux_dx, x, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
    dux_dyy_g = torch.autograd.grad(dux_dy, y, grad_outputs=ones, create_graph=True, allow_unused=True)[0]

    duy_dxx_g = torch.autograd.grad(duy_dx, x, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
    duy_dyy_g = torch.autograd.grad(duy_dy, y, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
    
    dux_dxx = dux_dxx_g if dux_dxx_g is not None else torch.zeros_like(ux)
    dux_dyy = dux_dyy_g if dux_dyy_g is not None else torch.zeros_like(ux)
    duy_dxx = duy_dxx_g if duy_dxx_g is not None else torch.zeros_like(ux)
    duy_dyy = duy_dyy_g if duy_dyy_g is not None else torch.zeros_like(ux)
    
    # =========================================================
    # CORRECCIÓN FÍSICA: Llevamos todo a unidades reales
    # =========================================================
    
    # 1. Recuperamos las velocidades reales
    ux_real = (ux - min_ux) / sux
    uy_real = (uy - min_uy) / suy
    
    # 2. Corregimos las primeras derivadas con la regla de la cadena
    dux_dx_real = dux_dx * (sx / sux)
    dux_dy_real = dux_dy * (sy / sux)
    duy_dx_real = duy_dx * (sx / suy)
    duy_dy_real = duy_dy * (sy / suy)
    
    # 3. Corregimos las segundas derivadas
    dux_dxx_real = dux_dxx * ((sx**2) / sux)
    dux_dyy_real = dux_dyy * ((sy**2) / sux)
    duy_dxx_real = duy_dxx * ((sx**2) / suy)
    duy_dyy_real = duy_dyy * ((sy**2) / suy)

    # =========================================================
    # Evaluar los residuos de Burgers con los valores físicos
    # =========================================================
    residuo_x = ux_real * dux_dx_real + uy_real * dux_dy_real - nu * (dux_dxx_real + dux_dyy_real)
    residuo_y = ux_real * duy_dx_real + uy_real * duy_dy_real - nu * (duy_dxx_real + duy_dyy_real)
    
    # MSE de la física
    loss_physics = torch.mean(residuo_x**2) + torch.mean(residuo_y**2)
    
    return loss_physics