import torch

def calcular_loss(modelo, x, y):
    """    
    Args:
        modelo: Instancia de InversePINN.
        x: Tensor de coordenadas X (N, 1) con requires_grad=True.
        y: Tensor de coordenadas Y (N, 1) con requires_grad=True.
    """
    
    ux, uy = modelo(x, y)
    nu = modelo.obtener_viscosidad()
  
    ones = torch.ones_like(ux)
    
    # Primeras derivadas
    dux_g = torch.autograd.grad(ux, [x, y], grad_outputs=ones, create_graph=True)
    dux_dx = dux_g[0]
    dux_dy = dux_g[1]
    
    duy_g = torch.autograd.grad(uy, [x, y], grad_outputs=ones, create_graph=True)
    duy_dx = duy_g[0]
    duy_dy = duy_g[1]
    
    # laplacianos
    dux_dxx = torch.autograd.grad(dux_dx, x, grad_outputs=ones, create_graph=True)[0]
    dux_dyy = torch.autograd.grad(dux_dy, y, grad_outputs=ones, create_graph=True)[0]

    duy_dxx = torch.autograd.grad(duy_dx, x, grad_outputs=ones, create_graph=True)[0]
    duy_dyy = torch.autograd.grad(duy_dy, y, grad_outputs=ones, create_graph=True)[0]
    
    # evaluar los residuos de las ecuaciones (deben tender a 0)
    residuo_x = ux * dux_dx + uy * dux_dy - nu * (dux_dxx + dux_dyy)
    residuo_y = ux * duy_dx + uy * duy_dy - nu * (duy_dxx + duy_dyy)
    
    # MSE 
    loss_physics = torch.mean(residuo_x**2) + torch.mean(residuo_y**2)
    
    return loss_physics