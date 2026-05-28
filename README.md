# InversePINN-Airfoils
Descubrimiento de viscosidad turbulenta y parámetros efectivos en perfiles alares 2D mediante Inverse PINNs

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):
    """
    Multilayer perceptron (MLP) // Perceptrón Multicapa .

    Esta clase define una red neuronal feedforward con múltiples capas ocultas
    lineales, funciones de activación tangente hiperbólica en  las capas ocultas
    y una salida lineal. Esta versión tiene como diferencia que existen
    parámetros entrenables que forman parte de la física del problema.

    Args:
        sizes (lista): Lista de enteros que especifica el número de neuronas en
        cada capa. El primer elemento debe coincidir con la dimensión de entrada
        y el último con la dimensión de salida.

    Atributos:
        -capas (torch.nn.ModuleList): Lista que contiene las capas lineales del MLP.
        -w0 (torch.nn.Parameter): Parámetro entrenable que representa la frecuencia natural del oscilador.
        -d (torch.nn.Parameter): Parámetro entrenable que representa el coeficiente de amortiguamiento.

    Métodos:
        forward(x): Realiza una pasada hacia adelante a través de la red MLP.

    """
    def __init__(self,sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.alp = torch.nn.Parameter(data=torch.Tensor([0.1]), requires_grad=True)
        self.Gam = torch.nn.Parameter(data=torch.Tensor([10]), requires_grad=True)
        self.r02 = torch.nn.Parameter(data=torch.Tensor([1]), requires_grad=True)
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y

pinn = MLP([1, 32, 32, 2])

optimizer = torch.optim.Adam(pinn.parameters(), lr=3e-4)

t_train = torch.tensor(t_data, dtype=torch.float32).view(-1,1).requires_grad_(True)
pos_data = torch.tensor(np.stack([x_data, y_data], axis=1), dtype=torch.float32)

#bucle de entrenamiento
epochs = 20000
for epoch in range(epochs):
  optimizer.zero_grad()

  #prediccion de la red
  pred_pos = pinn(t_train)
  x_p = pred_pos[:, 0:1] # Mantenemos la forma (N, 1)
  y_p = pred_pos[:, 1:2]


  loss_data = torch.mean((pred_pos - pos_data)**2)

  dx_dt = torch.autograd.grad(x_p, t_train, torch.ones_like(x_p), create_graph=True)[0]
  dy_dt = torch.autograd.grad(y_p, t_train, torch.ones_like(y_p), create_graph=True)[0]


  v_pinn = torch.cat([dx_dt, dy_dt], dim=1)

  r2 = x_p**2 + y_p**2 + 1e-8 # Un epsilon pequeño para evitar divisiones por cero
  term_circ = (pinn.Gam / (2 * np.pi * r2)) * (1 - torch.exp(-r2 / (pinn.r02)))

  u_x = -pinn.alp * x_p - y_p * term_circ
  u_y = -pinn.alp * y_p + x_p * term_circ

  u_burgers = torch.cat([u_x, u_y], dim=1)

# Loss de física
  loss_physics = torch.mean((v_pinn - u_burgers)**2)

  loss = loss_data + 1e-3 * loss_physics # aca ajustamos el peso

  loss.backward()
  optimizer.step()

  if epoch % 2000 == 0:
     print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Alp: {pinn.alp.item():.4f}, Gam: {pinn.Gam.item():.4f}, r02: {pinn.r02.item():.4f}")

#graficamos
t_eval = torch.linspace(0, float(t_data.max()), 200).view(-1, 1)
with torch.no_grad():
    final_pred = pinn(t_eval).numpy()

plt.figure(figsize=(8,6))
plt.plot(x_data, y_data, 'ko', label='Datos exp')
plt.plot(final_pred[:,0], final_pred[:,1], 'r-', label='Predicción PINN')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.legend()
#plt.title('!!')
plt.show()

