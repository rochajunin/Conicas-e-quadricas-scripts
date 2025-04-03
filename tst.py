# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt

# # Definir variáveis
# x, y = sp.symbols('x y')

# # Exemplo de equação da cônica rotacionada
# A, B, C, D, E, F = 3, 2, 2, -4, -5, -6  # Coeficientes da equação geral Ax² + Bxy + Cy² + Dx + Ey + F = 0
# equacao = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F

# # Gerar os pontos para o gráfico
# f_lambdified = sp.lambdify((x, y), equacao, 'numpy')

# X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
# Z = f_lambdified(X, Y)

# # Plotar o gráfico
# plt.contour(X, Y, Z, levels=[0], colors='blue')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Gráfico da Cônica Rotacionada")
# plt.grid()
# plt.show()




# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Definir variáveis
# x, y, z = sp.symbols('x y z')

# # Definir os coeficientes da equação geral
# A, B, C, D, E, F, G, H, I, J = 2,1,3,1,2,-1,0,0,0,-10  # Exemplo de hiperboloide

# # Construir a equação quádica
# equacao = A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J

# # Converter a equação simbólica para uma função NumPy
# f_lambdified = sp.lambdify((x, y, z), equacao, 'numpy')

# # Criar a malha de pontos para X e Y
# X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

# # Resolver Z implicitamente: z = sqrt(-(Ax² + By² + ... + J) / C)
# Z_positive = np.sqrt(-(A*X**2 + B*Y**2 + D*X*Y + G*X + H*Y + J) / C)
# Z_negative = -Z_positive  # Reflexo para obter os dois lados

# # Criar figura 3D
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plotar superfície
# ax.plot_surface(X, Y, Z_positive, color='b', alpha=0.6)
# ax.plot_surface(X, Y, Z_negative, color='b', alpha=0.6)

# # Configuração dos eixos
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Superfície Quádrica (Exemplo: Hiperboloide)")

# plt.show()
