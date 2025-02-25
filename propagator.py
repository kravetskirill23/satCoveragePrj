import numpy as np
from rk4 import *

earthRadius = 6378135      # Экваториальный радиус Земли [m]
earthGM = 3.986004415e+14  # Гравитационный параметр Земли [m3/s2]
earthJ2 = 1.082626e-3      # Вторая зональная гармоника геопотенциала

def orb2Cart(a, ecc, trueAnomaly, raan, inc, aop):
    mu = earthGM
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
    rotAop = np.array([[np.cos(aop), -np.sin(aop), 0],
                       [np.sin(aop), np.cos(aop), 0],
                       [0, 0, 1]])

    rVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(rVecOSC)))
    velVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(velVecOSC)))
    return np.concatenate((rVecISC, velVecISC), axis=None)

def cart2Orb(stateRV):

    mu = earthGM
    rVec = stateRV[0:3] # radius vector in ISC
    r = np.linalg.norm(rVec)

    vVec = stateRV[3:] # velocity vector in ISC
    v = np.linalg.norm(vVec)

    cVec = np.cross(rVec, vVec) # moment of the pulse vector
    c = np.linalg.norm(cVec)

    fVec = np.cross(vVec, cVec) - mu * rVec / r # Laplace vector
    f = np.linalg.norm(fVec)

    p = c**2 / mu # orbit parameter

    ecc =  f / mu # eccentricity

    a = p / (1 - ecc**2) # large semi-axis

    zAxis = np.array([0, 0, 1.0], dtype=float)

    inc = np.rad2deg(np.acos(np.dot(cVec, zAxis)/ c)) # inclination
    lVec = np.cross(zAxis, cVec)

    raan = np.rad2deg(np.arctan2(lVec[1], lVec[0])) # longitude of the ascending node

    clCross = np.cross(cVec, lVec)
    clVec  = clCross / np.linalg.norm(clCross)
    lVecNorm = lVec / np.linalg.norm(lVec)

    eccVec = 1 / mu * (np.cross(vVec, cVec) - mu * (rVec / r))
    # aop = (np.arctan2(eccVec[1], eccVec[0]) - raan) % (2 * np.pi) if ecc > 1e-8 else None  # argument of pericenter else NaN
    aop = np.rad2deg(np.arctan2(np.dot(clVec, fVec), np.dot(lVecNorm, fVec)) - np.pi/6)
    trueAnomaly = np.rad2deg((np.arccos(np.dot(eccVec/ecc, rVec/r))) % (2 * np.pi)) if ecc > 1e-8 else None
    
    return np.array([inc, raan, a, ecc, aop, trueAnomaly, c, f])

def propagateOrbit(a, ecc, trueAnomaly, raan, inc, aop, t, dt):
    
    state0 = orb2Cart(a, ecc, trueAnomaly, raan, inc, aop)
    
    twoBodyModel = lambda stateVec: np.concatenate((stateVec[3:], -earthGM * stateVec[0:3] / np.linalg.norm(stateVec[0:3])**3),axis=None)
    stateRVarr = RK4Model(state0, t, dt, twoBodyModel)
    
    return stateRVarr

# x = stateRVarr[:, 0]
# y = stateRVarr[:, 1]
# z = stateRVarr[:, 2]

