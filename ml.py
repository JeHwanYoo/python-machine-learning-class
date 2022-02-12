import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class ML(metaclass=ABCMeta):
  def __init__(self, x, t):
    self.W = np.random.rand(x.shape[1], 1)
    self.b = np.random.rand(1)
    self.x = x
    self.t = t

  def reset(self, x=None, t=None):
    if not x:
      self.W = np.random.rand(self.x.shape)
    else:
      self.W = np.random.rand(x.shape)
      self.x = x
    self.b = np.random.rand(1)
    self.t = t if self else self.t
  
  @abstractmethod
  def loss_val(self):
    pass

  def train(self, loss_val, epoch, learning_rate, debug_step=None):
    if not debug_step:
        debug_step = int(epoch * 0.1)
          
    for step in range(epoch):
      self.W -= learning_rate * ML.derivative(lambda x: loss_val(), self.W)
      self.b -= learning_rate * ML.derivative(lambda x: loss_val(), self.b)
      if step % debug_step == 0:
        print('step = ', step, 'loss value = ', loss_val())

    return (self.W, self.b)
  
  @abstractmethod
  def predict(self, x):
    pass

  @staticmethod
  def derivative(f, x, dx=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + dx
        fx1 = f(x)
        x[idx] = tmp_val - dx
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * dx)
        x[idx] = tmp_val
        it.iternext()
    return grad

  @staticmethod
  def sigmoid(z):
    return 1 / (1+np.exp(z))

class LinearRegressionMSE(ML):
  def loss_val(self):
    y = np.dot(self.x, self.W) + self.b
    return np.sum((self.t-y)**2) / len(self.x)
  
  def train(self, epoch, learning_rate, debug_step=None):
    return super().train(self.loss_val, epoch, learning_rate, debug_step)
  
  def predict(self, x):
    return np.dot(x, self.W) + self.b

class BinaryClassification(ML):
  def __init__(self, x, t, activate=ML.sigmoid):
    super().__init__(x, t)
    self.activate = activate

  def loss_val(self):
    delta = 1e-7
    z = np.dot(self.x, self.W) + self.b
    y = self.activate(z)
    return -np.sum(self.t*np.log(y + delta) + (1-self.t) * np.log((1 - y) + delta))

  def train(self, epoch, learning_rate, debug_step=None):
    return super().train(self.loss_val, epoch, learning_rate, debug_step)

  def predict(self, x):
    z = np.dot(x, self.W) + self.b
    return self.activate(z)

class MulticlassClassification(BinaryClassification):
  def predict(self, x):
    z = np.dot(x, self.W) + self.b
    y = self.activate(z)
    return (y, np.argmax(y))
