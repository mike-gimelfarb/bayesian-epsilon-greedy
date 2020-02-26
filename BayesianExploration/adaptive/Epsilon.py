from abc import ABC, abstractmethod
from collections import defaultdict
import math
from scipy.stats import t as tdist

from adaptive.Average import Average
from adaptive.Beta import Beta


class Epsilon(ABC):
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def get_epsilon(self, state):
        pass
    
    @abstractmethod
    def update_from_experts(self, state, data):
        pass
    
    @abstractmethod
    def update_end_of_episode(self, episode):
        pass


class Fixed(Epsilon):
    
    def __init__(self, value):
        self.value = value
        
    def clear(self):
        pass
    
    def get_epsilon(self, state):
        return self.value
    
    def update_from_experts(self, state, data):
        pass
    
    def update_end_of_episode(self, episode):
        pass
    

class ExponentialDecay(Epsilon):
    
    def __init__(self, initial, episode_decay, step_decay=1.0):
        self.initial = initial
        self.episode_decay = episode_decay
        self.step_decay = step_decay
        self.epsilon = self.initial
    
    def clear(self):
        self.epsilon = self.initial
    
    def get_epsilon(self, state):
        return self.epsilon
    
    def update_from_experts(self, state, data):
        self.epsilon *= self.step_decay
    
    def update_end_of_episode(self, episode):
        self.epsilon *= self.episode_decay


class PowerLawDecay(Epsilon):
    
    def __init__(self, initial, power):
        self.initial = initial
        self.power = power
            
    def clear(self):
        self.epsilon = self.initial
    
    def get_epsilon(self, state):
        return self.epsilon
    
    def update_from_experts(self, state, data):
        pass
    
    def update_end_of_episode(self, episode):
        self.epsilon = self.initial / (episode + 1.) ** self.power

    
class BMC(Epsilon):
    
    def __init__(self, alpha=1.0, beta=1.0, sigma_sq=None):
        self.alpha, self.beta = alpha, beta
        self.sigma_sq = sigma_sq
        
    def clear(self):
        self.stat = Average()
        self.post = Beta(self.alpha, self.beta)
    
    def get_epsilon(self, state):
        post = self.post
        return post.alpha / (post.alpha + post.beta)

    def update_from_experts(self, state, data):
        G_Q, G_U = data[0], data[1]
        epsilon = self.get_epsilon(state)
        G = (1.0 - epsilon) * G_Q + epsilon * G_U
        if self.sigma_sq is None:
            var = self.stat.update(G) 
            if var <= 0.0:
                return
        else:
            var = self.sigma_sq
        normalizer = math.log(2.0 * math.pi * var)
        phi_q = math.exp(-0.5 * ((G - G_Q) * (G - G_Q) / var + normalizer))
        phi_u = math.exp(-0.5 * ((G - G_U) * (G - G_U) / var + normalizer))
        self.post.update(phi_u, phi_q)
    
    def update_end_of_episode(self, episode):
        pass


class BMCRobust(Epsilon):
    
    def __init__(self, mu, tau, a, b, alpha=1.0, beta=1.0):
        self.mu0, self.tau0, self.a0, self.b0 = mu, tau, a, b
        self.alpha, self.beta = alpha, beta
        
    def clear(self):
        self.stat = Average()
        self.post = Beta(self.alpha, self.beta)
    
    def get_epsilon(self, state):
        post = self.post
        return post.alpha / (post.alpha + post.beta)

    def update_from_experts(self, state, data):
        
        # compute return
        G_Q, G_U = data[0], data[1]
        epsilon = self.get_epsilon(state)
        G = (1.0 - epsilon) * G_Q + epsilon * G_U
        
        # update mu-hat and sigma^2-hat
        self.stat.update(G)
        mu, sigma2, t = self.stat.mean, self.stat.var, self.stat.count
        
        # update a_t and b_t
        a = self.a0 + t / 2
        b = self.b0 + t / 2 * sigma2 + t / 2 * (self.tau0 / (self.tau0 + t)) * (mu - self.mu0) * (mu - self.mu0)
        
        # compute e_t
        scale = (b / a) ** 0.5
        e_u = tdist.pdf(G, df=2.0 * a, loc=G_U, scale=scale)
        e_q = tdist.pdf(G, df=2.0 * a, loc=G_Q, scale=scale)
        
        # update posterior
        self.post.update(e_u, e_q)
    
    def update_end_of_episode(self, episode):
        pass


class VDBE(Epsilon):
    
    def __init__(self, initial, delta, sigma, transform=None):
        self.initial = initial
        self.delta, self.sigma = delta, sigma
        self.transform = transform
        
    def clear(self):
        self.epsilon = defaultdict(lambda: self.initial)
    
    def get_epsilon(self, state):
        if self.transform is not None:
            state = self.transform(state)
        return self.epsilon[state]
    
    def update_from_experts(self, state, data):
        td_error = data[2]
        coeff = math.exp(-abs(td_error) / self.sigma)
        f = (1.0 - coeff) / (1.0 + coeff)
        if self.transform is not None:
            state = self.transform(state)
        self.epsilon[state] = self.delta * f + (1.0 - self.delta) * self.epsilon[state]
    
    def update_end_of_episode(self, episode):
        pass
