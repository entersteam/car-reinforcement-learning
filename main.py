import numpy as np

ACC = 0.1

rotation_rad = 0.1

def rotation_matrix(rad):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

def normalize_vector(v1):
    return v1/ np.linalg.norm(v1)

clockwise_rotation_matrix = rotation_matrix(rotation_rad)
counter_clockwise_rotation_matrix = rotation_matrix(-rotation_rad)

class car:
    def __init__(self, position, direction = np.zeros((2), dtype = float), *args, **kwargs) -> None:
        self.position = position
        self.direction = normalize_vector(direction)
        self.velocity = np.zeros((2), dtype = float)
        
        self.distance = 0
        
        self.rotate_func = [self.turn_right, self.turn_left, self.do_nothing]
        self.accel_func = [self.acceleration, self.deceleration, self.do_nothing]
        
        if 'parent' in kwargs:
            self.weights_rotate = kwargs['parent'].weights_rotate + np.random.normal(0, 1, (3,5))
            self.biases_rotate = kwargs['parent'].biases_rotate + np.random.normal(0, 1, 3)
            self.weights_accel = kwargs['parent'].weights_accel + np.random.normal(0, 1, (3,5))
            self.biases_accel = kwargs['parent'].biases_accel + np.random.normal(0, 1, 3)
        else:
            self.weights_rotate = np.random.normal(0, 1, (3,5))
            self.biases_rotate = np.random.normal(0, 1, 3)
            self.weights_accel = np.random.normal(0, 1, (3,5))
            self.biases_accel = np.random.normal(0, 1, 3)
        
        self.distances = np.zeros((5), dtype = float)
            
    def update(self):
        self.values_rotate = (self.weights_rotate @ self.distances) + self.biases_rotate
        self.values_accel = (self.weights_accel @ self.distances) + self.biases_accel
        
        self.rotate_func[np.argmax(self.values_rotate)]()
        self.accel_func[np.argmax(self.values_accel)]()
        
        self.position += self.velocity
        self.distance += np.linalg.norm(self.velocity)
            
    def acceleration(self):
        self.velocity += self.direction * ACC
        
    def deceleration(self):
        self.velocity -= self.direction * ACC
        
    def turn_right(self):
        self.direction = clockwise_rotation_matrix @ self.direction
        
    def turn_left(self):
        self. direction = counter_clockwise_rotation_matrix @ self.direction
        
    def do_nothing():
        pass
        
    def normalize_direction(self):
        self.direction = normalize_vector(self.direction)
        
class envirnoment:
    def __init__(self, points : list) -> None:
        self.roads = np.array((points, points[1:]+points[:1]))
        self.roads = np.rot90(self.roads, 3)
        print(self.roads)
        
        self.init_position = self.roads[0,0]
        self.init_direction = normalize_vector(self.roads[0,1] - self.roads[0,0])
        self.cars = []
        for i in range(20):
            self.cars.append(car(self.init_position, self.init_direction))
            

if __name__ == "__main__":
    envirnoment([[1,2],[2,3],[3,4],[4,5]])