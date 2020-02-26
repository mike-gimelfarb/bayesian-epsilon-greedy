class Average:

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean, self.m2, self.var, self.count = 0.0, 0.0, 0.0, 0

    def update(self, point):
        self.count += 1
        count = self.count
        delta = point - self.mean
        self.mean += delta / count
        self.m2 += delta * (point - self.mean)
        if count > 1:
            self.var = self.m2 / (count - 1.0)
        return self.var
