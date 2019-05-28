import numpy as np 

class OrnsteinUhlembeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x 

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise with parameter (mu:{0}, sigma:{1})'.format(self.mu, self.sigma)

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt 
#     model = OrnsteinUhlembeckActionNoise(mu=np.array([0.3]), x0=np.array([0]))
#     res = [0]
#     model.reset()
#     for _ in range(1000):
#         res.append(model.__call__())
#     plt.plot(res)
#     plt.show()
        