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

def intersection_distance(seg1_start, seg1_end, seg2_start, seg2_end):
    def direction(p1, p2, p3):
        return (p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])

    def on_segment(p1, p2, p3):
        return min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1])

    d1 = direction(seg2_start, seg2_end, seg1_start)
    d2 = direction(seg2_start, seg2_end, seg1_end)
    d3 = direction(seg1_start, seg1_end, seg2_start)
    d4 = direction(seg1_start, seg1_end, seg2_end)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        # Segments intersect
        intersection_x = ((seg1_start[0] * d2 - seg1_end[0] * d1) * (seg2_end[0] - seg2_start[0]) - (seg1_start[0] - seg1_end[0]) * (seg2_end[0] * d3 - seg2_start[0] * d4)) / ((d2 - d1) * (seg2_end[0] - seg2_start[0]) - (seg1_start[0] - seg1_end[0]) * (d4 - d3))
        intersection_y = ((seg1_start[1] * d2 - seg1_end[1] * d1) * (seg2_end[1] - seg2_start[1]) - (seg1_start[1] - seg1_end[1]) * (seg2_end[1] * d3 - seg2_start[1] * d4)) / ((d2 - d1) * (seg2_end[1] - seg2_start[1]) - (seg1_start[1] - seg1_end[1]) * (d4 - d3))

        intersection_point = (intersection_x, intersection_y)

        if on_segment(seg1_start, seg1_end, intersection_point) and on_segment(seg2_start, seg2_end, intersection_point):
            # Calculate distance
            distance = ((intersection_point[0] - seg1_start[0]) ** 2 + (intersection_point[1] - seg1_start[1]) ** 2) ** 0.5
            return intersection_point, distance

    return None


    
class car:
    def __init__(self, position, sense_distance = 5, direction = np.zeros((2), dtype = float), *args, **kwargs) -> None:
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
        self.sense_distance = sense_distance
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
        return np.array([self.sense_distance * (matrix @ self.direction) + self.position for matrix in senselines_rotation_matrixs])
        
            
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
    def __init__(self, cars_num : int, points : list) -> None:
        
        self.roads = np.concatenate((np.array(points), np.array(points[1:]+points[:1])), axis = 1).reshape(-1,2,3)
        print(self.roads)
        
        self.init_position = self.roads[0,0,:2]
        self.init_direction = normalize_vector(self.roads[0,1,:2] - self.roads[0,0,:2])
        
        self.walls = self.create_walls()
        
        self.cars_n = cars_num
        
        self.cars = []
        for i in range(cars_num):
            self.cars.append(car(self.init_position, self.init_direction))
            
    def sensing_distance(self):
        for i in range(self.cars_n):
            sense_lines = self.cars[i].sense_lines
            distances = []
            for sense_line in sense_lines:
                close_distance = self.cars[i].sense_distance
                for wall in self.walls:
                    cross = intersection_distance(self.cars[i].position, sense_line, wall[0], wall[1])
                    if not (cross) or (cross[1] >= close_distance):
                        close_distance = cross[1]
                distances.append(close_distance)
            self.cars[i].distances = np.array(distances)
                        

            
    def create_walls(self):
        inside_wall_points = []
        outside_wall_points = []
        for i in range(len(self.roads)):
            r1 = self.roads[i-1]
            r2 = self.roads[i]
            half_angle = half_angle_vector(r1[1,:2] - r1[0,:2], r2[1,:2] - r2[0,:2])
            inside_wall_points.append(r2[0,:2] + half_angle * r2[0,2])
            outside_wall_points.append(r2[0,:2] + half_angle * -r2[0,2])
        inside_walls = [inside_wall_points, inside_wall_points[1:] + inside_wall_points[0]]
        outside_walls = [outside_wall_points, outside_wall_points[1:] + outside_wall_points[0]]
        return np.concatenate((inside_walls, outside_walls))

if __name__ == "__main__":
    envirnoment([[1,2],[2,3],[3,4],[4,5]])