import math

from matplotlib.pylab import angle



def calculate_distance(point1,point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    distance = math.sqrt(dx**2 + dy**2)
    return distance

def calculate_angle(point1, point2, point3):
    vector1 = (point1[0] - point2[0],point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    length1 = math.hypot(vector1[0], vector1[1])
    length2 = math.hypot(vector2[0], vector2[1])
    cos_theta = dot_product / (length1 * length2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.degrees(math.acos(cos_theta))
    return angle
