# Simulación Numérica de la Ecuación de Onda Bidimensional

Este repositorio contiene una implementación en Python de una simulación numérica y una solución analítica de la ecuación de onda en dos dimensiones, utilizando condiciones de frontera tipo Neumann.


![Simulación 2D](frames/animacion.gif)


## 📌 Descripción

El script simula la evolución temporal de una onda en un canal bidimensional, comparando la solución numérica con la solución analítica. Se utilizan:
- Representación visual de ambas soluciones
- Animaciones en 2D
- Cálculo del error absoluto entre ambas soluciones

## ▶️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt

## ▶️ Ejecución

Simplemente corre el script de simulación:

```bash
python wave2D.py
```

Los resultados se guardarán en la carpeta `frames/`, incluyendo una animación `.gif` del proceso y gráficos del error.

## 📊 Resultados

- Comparación visual entre solución numérica y analítica
- Gráfico de evolución del error máximo
- Mapa de calor del error absoluto

## 🛠️ Autor

- **Tu Nombre**
- [Tu perfil de GitHub](https://github.com/DanZarate)
