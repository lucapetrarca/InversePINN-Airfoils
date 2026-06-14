import torch
import torch.nn as nn

class InversePINN(nn.Module):
    def __init__(self, num_capas=4, num_neuronas=64):
        super(InversePINN, self).__init__()

        capas = []
        
        #Se reciben coordenadas (x, y)
        capas.append(nn.Linear(2, num_neuronas))
        capas.append(nn.Tanh()) #Suave para derivar 2 veces en burgers

        #Se agregan las capas ocultas
        for _ in range(num_capas - 1):
            capas.append(nn.Linear(num_neuronas, num_neuronas))
            capas.append(nn.Tanh())

        #Se devuelven velocidades (ux, uy)
        capas.append(nn.Linear(num_neuronas, 2))

        self.red = nn.Sequential(*capas)

        #Se define 'visc' como parámetro a optimizar.
        #Se usa log y luego exp para asegurar visc > 0
        valor_inicial = torch.log(torch.tensor([0.01], dtype=torch.float32))
        self.visc = nn.Parameter(valor_inicial)

    def forward(self, x, y):
        entradas = torch.cat([x, y], dim=1)
        
        #Se asignan los valores de entrada a la secuencia de la red definida
        salidas = self.red(entradas)
        
        #Se devuelven las predicciones de ux y uy
        ux_pred = salidas[:, 0:1]
        uy_pred = salidas[:, 1:2]
        
        return ux_pred, uy_pred

    def obtener_viscosidad(self):
        #Se devuelve e^(viscosidad obtenida) > 0
        return torch.exp(self.visc)