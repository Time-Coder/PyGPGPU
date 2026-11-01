import math

class Interval:

    empty = None
    universe = None

    def __init__(self, min_value: float = math.inf, max_value: float = -math.inf):
        self.min_value = min_value
        self.max_value = max_value

    def size(self)->float:
        return self.max_value - self.min_value

    def __contains__(self, x: float)->bool:
        return self.min_value <= x <= self.max_value

    def surrounds(self, x: float)->bool:
        return self.min_value < x < self.max_value

    def clamp(self, x: float) -> float:
        if x < self.min_value:
            return self.min_value

        if x > self.max_value:
            return self.max_value

        return x

Interval.empty = Interval(math.inf, -math.inf)
Interval.universe = Interval(-math.inf, math.inf)