import torch

def calcular_loss(modelo, x, y):
    ux, uy = modelo(x, y)
    nu_adimensional = modelo.obtener_viscosidad() 
    
    ones = torch.ones_like(ux)
    
    #Primeras Derivadas
    dux_dx, dux_dy = torch.autograd.grad(ux, [x, y], grad_outputs=ones, create_graph=True)
    duy_dx, duy_dy = torch.autograd.grad(uy, [x, y], grad_outputs=ones, create_graph=True)
    
    #Laplacianos (Derivadas segundas)
    dux_dxx = torch.autograd.grad(dux_dx, x, grad_outputs=ones, create_graph=True)[0]
    dux_dyy = torch.autograd.grad(dux_dy, y, grad_outputs=ones, create_graph=True)[0]
    
    duy_dxx = torch.autograd.grad(duy_dx, x, grad_outputs=ones, create_graph=True)[0]
    duy_dyy = torch.autograd.grad(duy_dy, y, grad_outputs=ones, create_graph=True)[0]
    
    #Burgers Adimensional Directo
    residuo_x = ux * dux_dx + uy * dux_dy - nu_adimensional * (dux_dxx + dux_dyy)
    residuo_y = ux * duy_dx + uy * duy_dy - nu_adimensional * (duy_dxx + duy_dyy)
    
    loss_physics = torch.mean(residuo_x*2) + torch.mean(residuo_y*2)
    return loss_physics
