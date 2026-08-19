import matplotlib.pyplot as plt

with open("House Price Prediction Dataset.csv", encoding="utf-8") as archivo:
    filas = archivo.read().splitlines()[1:]

x = [float(fila.split(",")[2]) for fila in filas]  # Bedrooms
y = [float(fila.split(",")[9]) for fila in filas]  # Price

corte = int(len(x) * 0.70)
x_entrenamiento, x_prueba = x[:corte], x[corte:]
y_entrenamiento, y_prueba = y[:corte], y[corte:]

n = len(x_entrenamiento)
promedio_x = sum(x_entrenamiento) / n
promedio_y = sum(y_entrenamiento) / n

numerador = sum((x_entrenamiento[i] - promedio_x) * (y_entrenamiento[i] - promedio_y) for i in range(n))
denominador = sum((valor - promedio_x) ** 2 for valor in x_entrenamiento)
pendiente = numerador / denominador
intercepto = promedio_y - pendiente * promedio_x

bedrooms = float(input("Numero de bedrooms: "))
precio = intercepto + pendiente * bedrooms

print(f"Entrenamiento: {len(x_entrenamiento)} | Prueba: {len(x_prueba)}")
print(f"Precio estimado: {precio:.2f}")

extremos = [min(min(x), bedrooms), max(max(x), bedrooms)]
recta = [intercepto + pendiente * valor for valor in extremos]

plt.scatter(x_entrenamiento, y_entrenamiento, alpha=0.25, label="Entrenamiento")
plt.scatter(x_prueba, y_prueba, alpha=0.25, label="Prueba")
plt.plot(extremos, recta, color="red", label="Modelo")
plt.scatter(bedrooms, precio, color="black", s=100, label="Prediccion")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Prediccion del precio de una casa")
plt.legend()
plt.show()