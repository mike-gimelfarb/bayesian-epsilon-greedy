class Beta:
    
    def __init__(self, alpha0, beta0):
        self.alpha = alpha0
        self.beta = beta0
    
    def update(self, expert1, expert2):
        alpha, beta = self.alpha, self.beta
        mean = expert1 * alpha + expert2 * beta
        if mean <= 0.0:
            return
        m = alpha / (alpha + beta + 1.) * (expert1 * (alpha + 1.) + expert2 * beta) / mean
        s = alpha / (alpha + beta + 1.) * (alpha + 1.) / (alpha + beta + 2.) * \
            (expert1 * (alpha + 2.) + expert2 * beta) / mean
        r = (m - s) / (s - m * m)
        self.alpha, self.beta = m * r, (1. - m) * r
