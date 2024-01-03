# Fractional Fourier Transform implementation
# See http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/fourint.pdf
# Bailey, Swartztrauber "A Fast Method for the Numerical Evaluation of Continuous Fourier and Laplace Transforms"

from numpy import pi, exp, sqrt
import numpy as np
import scipy as sp
from scipy.fft import fftn, ifftn, fft, ifft
import matplotlib.pyplot as plt
import multiprocessing as mp

ipi = pi*1j

def compute_z(m, alpha):
    return np.concatenate((np.array([exp(ipi * i ** 2 * alpha) for i in range(m)]),
                    (np.array([exp(ipi * (i - m) ** 2 * alpha) for i in range(m)]))))

class Transformer:
    def __init__(self, data, data_intervals, output_intervals=()):
        #data interval should be tuple with resolutions for each variable

        self.data = data
        self.m = self.data.shape

        self.beta = np.array(data_intervals)
        self.gamma = np.array(output_intervals or data_intervals)
        if len(self.beta) != len(self.gamma) or len(data.shape) != len(self.beta):
            raise ValueError("All input and output intervals must be specified!")
        self.delta = np.multiply(self.beta, self.gamma) / sqrt(2*pi)

        self.z = {m: compute_z(m, self.delta) for m in set(self.m)}
        self.iz = {m: compute_z(m, -self.delta) for m in set(self.m)}

        #self.worker_pool = mp.pool.Pool(8)

    def cft_1d(self, x: np.ndarray, axis: int):
        delta = self.delta[axis]
        m = self.m[axis]
        transform = self.frft(np.array([x[j] * exp(ipi * j * m * delta) for j in range(m)]), m, delta, axis)
        return self.beta[axis] * np.array([exp(ipi * (k - m / 2) * m * delta) * transform[k] for k in range(m)])
    def icft_1d(self, x: np.ndarray, axis: int):
        delta = -self.delta[axis]
        m = self.m[axis]
        transform = self.ifrft(np.array([x[j] * exp(ipi * j * m * delta) for j in range(m)]), m, delta, axis)
        return self.beta[axis] * np.array([exp(ipi * (k - m / 2) * m * delta) * transform[k] for k in range(m)])

    def frft(self, x: np.ndarray, m: int, alpha: float, axis:int):
        y = np.concatenate((np.array([x[j]*exp(-ipi*j**2*alpha) for j in range(m)]), np.zeros(m)))
        #print(y.shape, self.z[m][axis])
        transform = ifft(np.multiply(fft(y),fft(self.z[m][:,axis])))
        return np.array([exp(-ipi*k**2*alpha) * transform[k] for k in range(m)])
    def ifrft(self, x: np.ndarray, m: int, alpha: float, axis:int):
        y = np.concatenate((np.array([x[j]*exp(-ipi*j**2*alpha) for j in range(m)]), np.zeros(m)))
        transform = ifft(np.multiply(fft(y),fft(self.iz[m][:,axis])))
        return np.array([exp(-ipi*k**2*alpha) * transform[k] for k in range(m)])

    def cft(self, *axes):
        for axis in axes:
            print('Doing CFT on axis', axis)
            self.data = np.apply_along_axis(self.cft_1d, axis, self.data, axis)
        return self
    def icft(self, *axes):
        for axis in axes:
            print('Doing inverse CFT on axis', axis)
            self.data = np.apply_along_axis(self.icft_1d, axis, self.data, axis)
        self.data = self.data / (2 * np.pi)
        return self


def cft(data, axes=(0,1), intervals=(1,1)):
    return Transformer(data, intervals).cft(*axes).data
def icft(data, axes, intervals):
    return Transformer(data, intervals).icft(*axes).data


if __name__ == "__main__":
    resolution = 0.5#np.sqrt(2*pi)/256
    X = np.empty((200, 200))
    for (p, _) in np.ndenumerate(X):
        X[p] = np.dot(np.add(p, -100), np.add(p, -100))

    print(X)
    y1 = exp(-X / 1000) / np.sqrt(2 * pi)

    y2 = np.abs(Transformer(y1, data_intervals=1/sqrt(1000)).cft(0).cft(1).data)

    plt.imshow(y2, origin='lower', interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
    plt.colorbar()
    plt.show()

    plt.imshow(y1, origin='lower', interpolation='bilinear')  # , extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
    plt.colorbar()
    plt.show()