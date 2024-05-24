import numpy as np

ACC = 0.1

rotation_rad = 0.1

def rotation_matrix(rad):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

def normalize_vector(v1):
    return v1/ np.linalg.norm(v1)

senselines_rotation_matrixs = np.array([rotation_matrix(-np.pi/3),rotation_matrix(-np.pi/6),rotation_matrix(0),rotation_matrix(np.pi/6),rotation_matrix(np.pi/3)])

right_angnle_rotation_matrix = np.array([[0, -1], [1,0]])

clockwise_rotation_matrix = rotation_matrix(rotation_rad)
counter_clockwise_rotation_matrix = rotation_matrix(-rotation_rad)


def half_angle_vector(v1,v2): # 왼쪽방향 벡터
    return normalize_vector(right_angnle_rotation_matrix @ (normalize_vector(v1) + normalize_vector(v2)))
    
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
        
        self.sense_lines = self.create_senselines()
        
        self.distances = np.zeros((5), dtype = float)
            
    def update(self):
        self.values_rotate = (self.weights_rotate @ self.distances) + self.biases_rotate
        self.values_accel = (self.weights_accel @ self.distances) + self.biases_accel
        
        self.rotate_func[np.argmax(self.values_rotate)]()
        self.accel_func[np.argmax(self.values_accel)]()
        
        self.position += self.velocity
        self.distance += np.linalg.norm(self.velocity)
        
        self.sense_lines = self.create_senselines()
        
    def create_senselines(self):
        return np.array([5 * (matrix @ self.direction) + self.position for matrix in senselines_rotation_matrixs])
        
            
    def acceleration(self):
        self.velocity += self.direction * ACC
        
    def deceleration(self):
        self.velocity -= self.direction * ACC
        
    def turn_right(self):
        self.direction = normalize_vector(clockwise_rotation_matrix @ self.direction)
        
    def turn_left(self):
        self. direction = normalize_vector(counter_clockwise_rotation_matrix @ self.direction)
        
    def do_nothing():
        pass
        
class envirnoment:
    def __init__(self, points : list) -> None:
        
        self.roads = np.concatenate((np.array(points), np.array(points[1:]+points[:1])), axis = 1).reshape(-1,2,3)
        print(self.roads)
        
        self.init_position = self.roads[0,0,:2]
        self.init_direction = normalize_vector(self.roads[0,1,:2] - self.roads[0,0,:2])
        
        self.walls = np.array([])
        
        self.cars = []
        for i in range(20):
            self.cars.append(car(self.init_position, self.init_direction))
            
    def create_walls(self):
        for i in range(len(self.roads)):
            r1 = self.roads[i-1]
            r2 = self.roads[i]
            half_angle = half_angle_vector(r1[1,:2] - r1[0,:2], r2[1,:2] - r2[0,:2])
            

if __name__ == "__main__":
    envirnoment([[1,2],[2,3],[3,4],[4,5]])