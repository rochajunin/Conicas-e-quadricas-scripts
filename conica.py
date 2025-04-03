import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Definir variáveis
x, y = sp.symbols('x y')

# Exemplo de equação da cônica rotacionada
A, B, C, D, E, F = 3, 2, 2, -4, -5, -6  # Coeficientes da equação geral Ax² + Bxy + Cy² + Dx + Ey + F = 0
equacao = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F

# Gerar os pontos para o gráfico
f_lambdified = sp.lambdify((x, y), equacao, 'numpy')

X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
Z = f_lambdified(X, Y)

# Plotar o gráfico
plt.contour(X, Y, Z, levels=[0], colors='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gráfico da Cônica Rotacionada")
plt.grid()
plt.show()