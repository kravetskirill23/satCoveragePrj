import  numpy as np

# Решаем уравнение с помощью метода Рунге-Кутты четвертого порядка
def RK4Model(y0, t, dt, vecFunction):
    # y - список размерности size(y0) x size(t),
    # vecFunction - векторное Лямбда-выражение модели движения
    # где y0 - вектор начальных данных, например, size[r, v, q, omega] = 12
    y = np.array([y0 for i in range(len(t))], dtype=float)
    y[0] = y0 # начальное условие
    for i in range(1, len(t)):
        k1 = vecFunction(y[i-1])
        k2 = vecFunction(y[i-1] + dt/2 * k1)
        k3 = vecFunction(y[i-1] + dt/2 * k2)
        k4 = vecFunction(y[i-1] + dt * k3)

        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y # решение модели - ссписок из фазовых векторов в каждый момент времени t
