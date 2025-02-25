import numpy as np
from typing import NamedTuple

earthRadius = 6378135      # Экваториальный радиус Земли [m]
earthGM = 3.986004415e+14  # Гравитационный параметр Земли [m3/s2]
earthJ2 = 1.082626e-3      # Вторая зональная гармоника геопотенциала

class OrbitalElements(NamedTuple):
    altitude: float
    inclination: float
    raan: float
    trueAnomaly: float
    

class Satellite(OrbitalElements):
    
    def setName(self, name):
        self.satName = name
          
    def setOrbitalElems(self):
        self.ecc = 0
        self.sma = Const.earthRadius + self.altitude*1e3
        self.aop = 0
        
    def __orb2Cartesian__(self):
        a = earthRadius + self.altitude *1e3
        ecc = 0
        aop = 0
        trueAnomaly = np.deg2rad(self.trueAnomaly)
        mu = earthGM
        raan = np.deg2rad(self.raan)
        inc = np.deg2rad(self.inclination)
        p = a * (1 - ecc ** 2)
        r = p / (1 + ecc * np.cos(trueAnomaly))
        trueAnomalyDot = np.sqrt(p * mu) / r**2
        rDot = - p * ecc * trueAnomalyDot * np.sin(trueAnomaly) / (1 + ecc * np.cos(trueAnomaly)) ** 2
        rVecOSC = np.array([r * np.cos(trueAnomaly), r * np.sin(trueAnomaly), 0])
        vxOSC = rDot * np.cos(trueAnomaly) - r * trueAnomalyDot * np.sin(trueAnomaly)
        vyOSC = rDot * np.sin(trueAnomaly) + r * trueAnomalyDot * np.cos(trueAnomaly)
        velVecOSC = np.array([vxOSC, vyOSC, 0])

        rotRaan = np.array([[np.cos(raan), -np.sin(raan), 0],
                            [np.sin(raan), np.cos(raan), 0],
                            [0, 0, 1]])
        rotInc = np.array([[1, 0, 0],
                        [0, np.cos(inc), -np.sin(inc)],
                        [0, np.sin(inc), np.cos(inc)]])
        
        rVecISC = rotRaan.dot(rotInc.dot(rVecOSC))
        velVecISC = rotRaan.dot(rotInc.dot(velVecOSC))
        return np.concatenate((rVecISC, velVecISC), axis=None)

    def getAccelerationJ2(rVec):
        r = np.linalg.norm(rVec)

        cVec = np.cross(rVec, vVec) # moment of the pulse vector
        c = np.linalg.norm(cVec)

        zAxis = np.array([0, 0, 1.0], dtype=float)

        inc = np.acos(np.dot(cVec, zAxis)/ c) # inclination
        lVec = np.cross(zAxis, cVec)

        raan = np.arctan2(lVec[1], lVec[0]) # longitude of the ascending node

        vecRaan = np.array([np.cos(raan), np.sin(raan), 0.0])
        u = np.acos(np.dot(vecRaan, rVec)/r)
        
        epsilon = 3 / 2 * earthGM * earthRadius**2 * earthJ2
        
        accelOrb = epsilon / r**4 * np.array(
            [
                3 * np.sin(u)**2 * np.sin(inc)**2 - 1, \
                -np.sin(2 * u) * np.sin(inc)**2, \
                -np.sin(2 * inc) * np.sin(u)
            ])
        
        
        rotRaan = np.array([[np.cos(raan), -np.sin(raan), 0],
                            [np.sin(raan), np.cos(raan), 0],
                            [0, 0, 1]])
        rotInc = np.array([[1, 0, 0],
                        [0, np.cos(inc), -np.sin(inc)],
                        [0, np.sin(inc), np.cos(inc)]])
        
        acceleration = rotRaan.dot(rotInc.dot(accelOrb))
        return acceleration



    # Решаем уравнение с помощью метода Рунге-Кутты четвертого порядка
    def __RK4Model__(y0, t, dt, vecFunction):
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

    def propagateOrbit(self, t, dt, num):
        state0 = self.__orb2Cartesian__(self.altitude, self.ecc, self.trueAnomaly, self.raan, self.inclination, self.aop)
        
        twoBodyModel = lambda stateVec: np.concatenate((stateVec[3:], self.getAccelerationJ2(stateVec[0:3])),axis=None)
        stateRVarr = self.__RK4Model__(state0, t, dt, twoBodyModel)
            
        return stateRVarr
    
    def getInitialState(self, num):
        self.ecc = 0
        self.sma = earthRadius + self.altitude*1e3
        self.aop = 0
        state0 = self.__orb2Cartesian__()
            
        return state0         


class Walker(NamedTuple):
    inclination: float           # наклонение орбиты
    satsPerPlane: int            # число КА в каждой орбитальной плоскости группы
    planeCount: int              # число орбитальных плоскостей в группе
    phase: int                   # фазовый сдвиг по аргументу широты между КА в соседних плоскостях
    altitude: float              # высота орбиты
    maxRaan: float               # максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей)
    startRaan: float             # прямое восхождение восходящего узла для первой плоскости


class WalkerGroup(Walker):
    
    def getInitialState(self, num):
        trueAnomaly = (num - (num // self.planeCount)* self.satsPerPlane) * self.phase
        raan = self.startRaan + (num // self.planeCount) * ((self.maxRaan - self.startRaan)/self.planeCount)
        sat = Satellite(self.altitude, self.inclination, raan, trueAnomaly)
       
        return sat.getInitialState(num)
        
    
    def propagate(self, t, dt, num):
        trueAnomaly = (num - (num // planeCount)*satsPerPlane) * phase
        raan = startRaan + (num // planeCount) * ((maxRaan - startRaan)/planeCount)
        elements = OrbitalElements([self.altitude, self.inclination, raan, trueAnomaly])
        sat = Satellite(elements)
        return sat.propagateOrbit(t, dt)
    
    def getTotalSatCount(self):
        return satsPerPlane * planeCount

class Constellation:
    
    def __init__(self):
        self.groups = []
        
    def addWalkers(self, walkers):
        walkersCount = 0
       
        for i in range(len(walkers)):
           
            walkerSatCount = walkers[i][2] * walkers[i][3]
        
            for count in range(walkerSatCount):
                self.groups.append(WalkerGroup(*walkers[i][1:]))
                        
                    
            
    def addBackupSatellites(self, satellites):
        
        for i in range(len(satellites)):
            self.groups.append(Satellite(*satellites[i][1:]))  


    def getInitialState(self):
        
        constellationState = np.array([[0.0]*6 for i in range(len(self.groups))])
        
        for i in range(len(constellationState)):
            constellationState[i] = self.groups[i].getInitialState(i + 1)
        return constellationState            


