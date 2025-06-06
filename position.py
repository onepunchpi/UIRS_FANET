import math
from typing import Union


class Position:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def set_xyz(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def change_xyz(self, x: float = 0, y: float = 0, z: float = 0):
        self.set_xyz(self.x + x, self.y + y, self.z + z)

    def distance_to(self, point: Union['Position', tuple[float, float, float]]):
        """
        Вычисляет евклидово расстояние между двумя позициями.
        :param point: другая позиция
        :return: расстояние в метрах
        """

        if isinstance(point, Position):
            return math.sqrt(
                (self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2
            )
        elif isinstance(point, tuple):
            return math.sqrt(
                (self.x - point[0])**2 + (self.y - point[1])**2 + (self.z - point[2])**2
            )
        else:
            raise 'Неверный тип точки'

    def get_tuple(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"(x={self.x}, y={self.y}, z={self.z})"

