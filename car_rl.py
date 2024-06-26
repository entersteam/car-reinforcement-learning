import numpy as np
import cv2

ACC = 0.5

SPEED_LIMIT = 5

rotation_rad = 0.1

def rotation_matrix(rad):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

def normalize_vector(v1):
    d = np.linalg.norm(v1)
    return v1.copy()/ d

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
        A1 = seg1_end[1] - seg1_start[1]
        B1 = seg1_start[0] - seg1_end[0]
        C1 = A1 * seg1_start[0] + B1 * seg1_start[1]

        A2 = seg2_end[1] - seg2_start[1]
        B2 = seg2_start[0] - seg2_end[0]
        C2 = A2 * seg2_start[0] + B2 * seg2_start[1]

        determinant = A1 * B2 - A2 * B1

        if determinant != 0:
            intersection_x = (B2 * C1 - B1 * C2) / determinant
            intersection_y = (A1 * C2 - A2 * C1) / determinant

            intersection_point = (intersection_x, intersection_y)

            if on_segment(seg1_start, seg1_end, intersection_point) and on_segment(seg2_start, seg2_end, intersection_point):
                # Calculate distance
                distance = ((intersection_point[0] - seg1_start[0]) ** 2 + (intersection_point[1] - seg1_start[1]) ** 2) ** 0.5
                return intersection_point, distance

    return None


def point_line_distance(point, line_start, line_end):
    # 넘파이 배열로 변환
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # 점과 선분의 시작점 간의 벡터
    line_vector = line_end - line_start

    # 점과 선분 시작점 사이의 벡터
    point_to_start_vector = point - line_start

    # 점과 선분 끝점 사이의 벡터
    point_to_end_vector = point - line_end

    # 선분의 길이 제곱 계산
    line_length_squared = np.sum(line_vector ** 2)

    # 점과 선분 사이의 수직 거리 계산
    dot_product = np.dot(point_to_start_vector, line_vector)

    # 수직 거리가 선분의 시작점과 끝점 사이에 있는지 확인
    if dot_product <= 0:
        return np.linalg.norm(point_to_start_vector)

    if dot_product >= line_length_squared:
        return np.linalg.norm(point_to_end_vector)

    # 수직 거리가 선분 내에 있는 경우
    perpendicular_distance = np.linalg.norm(np.cross(point_to_start_vector, line_vector)) / np.linalg.norm(line_vector)
    return perpendicular_distance


class car:
    sense_distance = 128
    
    def __init__(self, position,  direction = np.zeros((2), dtype = float), *args, **kwargs) -> None:
        self.position = position.copy()
        self.direction = normalize_vector(direction.copy())
        self.velocity = np.zeros((2), dtype = float)

        self.distance = 0
        self.duration = 0

        self.rotate_func = [self.turn_right, self.turn_left, self.do_nothing]
        self.accel_func = [self.acceleration, self.deceleration]

        if 'parent' in kwargs:
            self.weights_rotate = kwargs['parent'].weights_rotate.copy() + (np.random.rand(3,5) - 0.5)
            self.biases_rotate = kwargs['parent'].biases_rotate.copy() + (np.random.rand(3) - 0.5)
            self.weights_accel = kwargs['parent'].weights_accel.copy() + (np.random.rand(5) - 0.5)
            self.biases_accel = kwargs['parent'].biases_accel.copy() + (np.random.rand(2) - 0.5)
            
            self.weights_rotate = np.maximum(0, np.minimum(self.weights_rotate, 1))
            self.biases_rotate = np.maximum(0, np.minimum(self.biases_rotate, 1))
            self.weights_accel = np.maximum(0, np.minimum(self.weights_accel, 1))
            self.biases_accel = np.maximum(0, np.minimum(self.biases_accel, 1))
        else:
            self.weights_rotate = np.random.rand(3,5)
            self.biases_rotate = np.random.rand(3)
            self.weights_accel = np.random.rand(5)
            self.biases_accel = np.random.rand(2)
        self.sense_lines = self.create_senselines()

        self.distances = np.zeros((5), dtype = float)


    def reset(self, position = np.zeros((2), dtype = float), direction = np.zeros((2), dtype = float), reset_parameter = True):

        self.position = np.float64(position.copy())
        self.direction = normalize_vector(direction.copy())
        self.velocity = np.zeros((2), dtype = float)

        self.distance = 0
        self.duration = 0
        if reset_parameter:
            self.weights_rotate = np.random.rand(3,5)
            self.biases_rotate = np.random.rand(3)
            self.weights_accel = np.random.rand(5)
            self.biases_accel = np.random.rand(2)

        self.sense_lines = self.create_senselines()

    def print(self):
        print(f'position : {self.position}')
        print(f'direction : {self.direction}')
        print(f'velocity : {self.velocity}')
        print(self.biases_accel)
        print('------------------------------------')

    def update(self):
        self.duration += 1


        self.values_rotate = (self.weights_rotate @ self.distances) + self.biases_rotate
        self.values_accel = (self.weights_accel @ self.distances) + self.biases_accel

        self.rotate_func[np.argmax(self.values_rotate)]()
        self.accel_func[np.argmax(self.values_accel)]()

        self.position += self.velocity.copy()
        self.distance += np.linalg.norm(self.velocity)

        self.sense_lines = self.create_senselines()

    def create_senselines(self):
        return np.array([(car.sense_distance * (matrix @ self.direction.copy()) + self.position ) for matrix in senselines_rotation_matrixs])


    def acceleration(self):
        self.velocity += self.direction.copy() * ACC
        if np.linalg.norm(self.velocity) > SPEED_LIMIT:
            self.velocity = normalize_vector(self.velocity) * SPEED_LIMIT

    def deceleration(self):
        self.velocity -= self.direction.copy() * ACC
        if np.linalg.norm(self.velocity) > SPEED_LIMIT:
            self.velocity = normalize_vector(self.velocity) * SPEED_LIMIT

    def turn_right(self):
        self.direction = normalize_vector(clockwise_rotation_matrix.copy() @ self.direction.copy())

    def turn_left(self):
        self.direction = normalize_vector(counter_clockwise_rotation_matrix.copy() @ self.direction.copy())

    def do_nothing(self):
        pass

class enviroment:
    def __init__(self, cars_num : int, points : list , width = 1600, height = 900 ,winname = 'car-rl') -> None:
        self.window_name = winname
        
        self.image_height = height
        self.image_width = width
        
        self.time = 0

        self.roads = np.concatenate((np.array(points), np.array(points[1:]+points[:1])), axis = 1).reshape(-1,2,3)

        self.checkpoints_n = len(points)
        self.checkpoints = np.array(points)
        self.cars_now_points = np.zeros((cars_num), dtype=int)

        self.init_position = np.float64(self.roads[0,0,:2])
        self.init_direction = normalize_vector(self.roads[0,1,:2] - self.roads[0,0,:2])

        self.walls = self.create_walls()
        
        self.road_map = np.full((self.image_height, self.image_width, 3), 255,dtype=np.uint8)
        print(self.walls)
        for wall in self.walls:
            cv2.line(self.road_map, np.intp(wall[0]), np.intp(wall[1]), (0,0,0), 1)

        self.cars_n = cars_num

        self.cars_activity = [True] * cars_num
        self.cars_done = [False] * cars_num

        self.cars = []
        for i in range(cars_num):
            self.cars.append(car(self.init_position, self.init_direction))

    def visualize(self):
        image = self.road_map.copy()
        for i, car in enumerate(self.cars):
            angle = np.degrees(np.arctan2(car.direction[1], car.direction[0]))

            # RotatedRect 객체 생성
            rotated_rect = ((car.position[0], car.position[1]), (24, 18), angle)

            # RotatedRect에서 4개의 꼭짓점 계산
            box = cv2.boxPoints(rotated_rect)
            box = np.intp(box)

            # 사각형 그리기
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                        
            # 중심에 숫자 표시
            number = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(number, font, font_scale, font_thickness)
            text_x = int(car.position[0]) - text_size[0] // 2
            text_y = int(car.position[1]) + text_size[1] // 2

            cv2.putText(image, number, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            for sense_line in car.sense_lines:
                cv2.line(image, np.intp(car.position), np.intp(sense_line), (125,125,125), 1)
            
        cv2.imshow(self.window_name, image)

    def train(self, n, infinite_learning = False):
        for i in range(n):
            self.Try()
            self.new_generation()

    def Try(self):
        while False in self.cars_done:
            self.time += 1
            self.sensing_distance()
            for i in range(self.cars_n):
                if self.cars_activity[i]:
                    self.cars[i].update()
                    checkpoint = self.checkpoints[(self.cars_now_points[i]%self.checkpoints_n)]
                    if np.linalg.norm(checkpoint[:2] - self.cars[i].position) <= checkpoint[2]:
                        self.cars_now_points[i] += 1
                        if self.cars_now_points[i] == self.checkpoints_n + 1:
                            self.cars_done[i] = True
            self.visualize()
            if cv2.waitKey(3) == 27:
                break
        

    def new_generation(self):
        print(len(self.cars), self.cars_now_points)
        
        n = int(self.cars_n/5)
        sorted_indexs = np.argsort(self.cars_now_points)
        print(sorted_indexs)
        
        selected_cars = []
        for i in range(n):
            selected_cars.append(self.cars[sorted_indexs[self.cars_n-1-i]])
        new_cars_generation = selected_cars
        for i in range(self.cars_n - n):
            child_car = car(self.init_position, self.init_direction, parent = selected_cars[i%n])
            new_cars_generation.append(child_car)
        self.cars = new_cars_generation
        self.env_reset(False)

    def env_reset(self, reset_parameter):
        self.cars_now_points = np.zeros((self.cars_n), dtype=int)
        self.cars_activity = [True] * self.cars_n
        self.cars_done = [False] * self.cars_n
        
        for i in range(self.cars_n):
            self.cars[i].reset(self.init_position, self.init_direction, reset_parameter)

    def sensing_distance(self):
        for i in range(self.cars_n):
            if self.cars_activity[i]:
                sense_lines = self.cars[i].sense_lines
                distances = []
                for sense_line in sense_lines:
                    close_distance = self.cars[i].sense_distance
                    for wall in self.walls:
                        if point_line_distance(self.cars[i].position.copy(), wall[0], wall[1]) < 15:
                            self.cars_activity[i] = False
                            self.cars_done[i] = True
                            
                            break
                        cross = intersection_distance(self.cars[i].position, sense_line, wall[0], wall[1])
                        if  (cross) and (cross[1] < close_distance):
                            close_distance = cross[1]
                    distances.append(close_distance)
                self.cars[i].distances = np.array(distances) / car.sense_distance



    def create_walls(self):
        inside_wall_points = []
        outside_wall_points = []
        for i in range(len(self.roads)):
            r1 = self.roads[i-1]
            r2 = self.roads[i]
            half_angle = half_angle_vector(r1[1,:2] - r1[0,:2], r2[1,:2] - r2[0,:2])
            inside_wall_points.append(r2[0,:2] + half_angle * r2[0,2])
            outside_wall_points.append(r2[0,:2] + half_angle * -r2[0,2])
        
        inside_walls = np.concatenate([inside_wall_points, inside_wall_points[1:] + [inside_wall_points[0]]], axis=1).reshape(-1,2,2)
        outside_walls = np.concatenate([outside_wall_points, outside_wall_points[1:] + [outside_wall_points[0]]], axis=1).reshape(-1,2,2)
        return np.concatenate((inside_walls, outside_walls))

if __name__ == "__main__":
    pass
