from position import Position
from signal_source import SignalSource, Channel, Transmitter, Modulation
import time
from typing import Union
from data_types import Frequency, Power

class Network:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node: 'Node'):
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            raise 'Нода с таким ID уже существует'

    def delete_node(self, node: Union['Node', int]):
        if isinstance(node, Node):
            del self.nodes[node.id]
        elif isinstance(node, int):
            del self.nodes[node]
        else:
            raise 'Неверный ID'

    def clear_nodes(self):
        self.nodes = {}

    def nodes_mapping(self):
        pass

    def get_track(self):
        pass


class Package:
    def __init__(self,
                 data: any,
                 package_id: int = None,
                 timestamp: float = None,
                 metadata: dict = None):
        """
        :param data: содержимое пакета (любой тип)
        :param package_id: уникальный идентификатор пакета (если нужно)
        :param timestamp: время создания/отправки пакета (если не указано — ставится текущее время)
        :param metadata: словарь с дополнительной информацией
        """
        self.data = data
        self.package_id = package_id
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<Package id={self.package_id} data={self.data} time={self.timestamp}>"


class Node:
    def __init__(self,
                 transmitter: Transmitter,
                 position: Position = None,
                 name: str = None):
        self.transmitter = transmitter
        if position is not None:
            self.transmitter.set_pos(position.x, position.y, position.z)
        self.name = name
        self.inbox = []

    def set_modulation(self, modulation: Modulation):
        self.transmitter.set_modulation(modulation)

    def set_channel(self, channel: Channel):
        self.transmitter.channel = channel

    @property
    def position(self) -> Position:
        return self.transmitter.position

    def move(self, x: float = 0, y: float = 0, z: float = 0):
        """Сдвигает позицию источника сигнала на заданные координаты"""
        self.transmitter.move(x, y, z)

    def move_to(self, x: float = None, y: float = None, z: float = None):
        """Двигает источник сигнала к заданным координатам"""
        self.transmitter.move_to(x, y, z)

    def set_pos(self, x: float = 0, y: float = 0, z: float = 0):
        self.transmitter.set_pos(x, y, z)

    def __repr__(self):
        return f"<Node id={self.name} \n    position={self.position}, \n    transmitter=\n{self.transmitter.__repr__()}>"



# class Node(Transmitter):
#     def __init__(
#         self,
#         name: str,
#         position: Position,
#         power: Power,
#         freq_range: tuple[Frequency, Frequency],
#         sensitivity: Power,
#         min_sir: Power,
#         current_channel: Frequency,
#         path_loss_exponent: float = 2.0
#     ):
#         """
#         :param name: str — уникальное имя или ID ноды
#         :param position: Position — положение узла
#         :param power: Power — выходная мощность передатчика узла
#         :param freq_range: (Frequency, Frequency) — допустимый диапазон частот
#         :param sensitivity: Power — порог приёма (минимальная мощность)
#         :param min_sir: Power — минимальное отношение сигнал/помеха для успешной передачи
#         :param current_channel: Frequency — текущая рабочая частота
#         :param path_loss_exponent: float — коэффициент затухания
#         """
#         super().__init__(
#             position=position,
#             power=power,
#             freq_range=freq_range,
#             sensitivity=sensitivity,
#             min_sir=min_sir,
#             current_channel=current_channel,
#             path_loss_exponent=path_loss_exponent
#         )
#         self.id = name
#         self.inbox = []  # для хранения полученных пакетов
#
#     def move(self, x: float = 0, y: float = 0, z: float = 0):
#         """Сдвигает позицию узла на заданные координаты"""
#         self.position.change_xyz(x, y, z)
#
#     def signal_quality_to(self, target: "Node", interference_power: float) -> float:
#         """Возвращает отношение сигнал/помехи (SIR) для передачи от этой ноды к целевой."""
#         signal_power = self.signal_at(target.position)
#
#         if interference_power == 0:
#             # Нет помех — идеальный сигнал
#             return float('inf')
#
#         return signal_power / interference_power
#
#     def can_deliver_to(self, target: "Node", interference_power: float, threshold: float = 0.01, min_sir: float = 1.0) -> bool:
#         """
#         Проверяет, может ли данный узел доставить пакет до цели с учётом интерференции и порогов.
#         :param target: целевая нода
#         :param interference_power: уровень помех на приёмнике
#         :param threshold: минимальная мощность сигнала, необходимая для приёма
#         :param min_sir: минимальное отношение сигнал/помеха (SIR) для успешной передачи
#         :return: True, если передача возможна
#         """
#         signal_power = self.signal_at(target.position)
#
#         if signal_power < threshold:
#             print(f"Node {self.id} сигнал слишком слабый для Node {target.id}: {signal_power:.6f}W < {threshold}W")
#             return False
#
#         sir = self.signal_quality_to(target, interference_power)
#         if sir < min_sir:
#             print(f"Передача от Node {self.id} к Node {target.id} прервана из-за помех. SIR={sir:.2f} < {min_sir}")
#             return False
#
#         return True
#
#     def attempt_receive(self, sender: 'Node', signal_power: float, interference_power: float, threshold: float = 0.01,
#                         min_sir: float = 1.0) -> bool:
#         """
#         Проверяет, может ли приёмник принять пакет от отправителя.
#         """
#         if signal_power < threshold:
#             print(
#                 f"Node {self.id} НЕ может принять: сигнал от Node {sender.id} слишком слабый: {signal_power:.6f}W < {threshold}W")
#             return False
#
#         if interference_power > 0:
#             sir = signal_power / interference_power
#         else:
#             sir = float('inf')
#
#         if sir < min_sir:
#             print(f"Node {self.id} НЕ может принять: помехи слишком сильны. SIR={sir:.2f} < {min_sir}")
#             return False
#
#         return True
#
#     def send(self, receiver: 'Node', package: 'Package', interference_power: float = 0.0, threshold: float = 0.01,
#              min_sir: float = 1.0) -> bool:
#         """
#         Пытается отправить пакет другой ноде с учётом интерференции.
#         """
#         if not self.can_deliver_to(receiver, interference_power, threshold, min_sir):
#             print(f"Node {self.id} не смог отправить пакет Node {receiver.id}")
#             return False
#
#         receiver.receive(package, self)
#         print(f"Node {self.id} успешно отправил пакет Node {receiver.id}")
#         return True
#
#     def receive(self, package: 'Package', sender: 'Node', interference_power: float = 0.0, threshold: float = 0.01,
#                 min_sir: float = 1.0):
#         """
#         Пытается принять пакет от другого узла, учитывая помехи.
#         """
#         signal_power = sender.signal_at(self.position)
#
#         if self.attempt_receive(sender, signal_power, interference_power, threshold, min_sir):
#             self.inbox.append((package, sender))
#             print(f"Node {self.id} ПРИНЯЛ пакет от Node {sender.id}")
#         else:
#             print(f"Node {self.id} ОТКЛОНИЛ пакет от Node {sender.id}")
#
#     def measure_link_quality(self, target: 'Node', interference_to_target: float = 0.0,
#                              interference_to_self: float = 0.0,
#                              threshold: float = 0.01, min_sir: float = 1.0) -> dict:
#         """
#         Двусторонняя оценка качества канала между этой нодой и целевой нодой.
#         """
#         # Прямое направление: self -> target
#         forward_signal = self.signal_at(target.position)
#         forward_sir = forward_signal / interference_to_target if interference_to_target > 0 else float('inf')
#         can_forward = self.can_deliver_to(target, interference_to_target, threshold, min_sir)
#
#         # Обратное направление: target -> self
#         reverse_signal = target.signal_at(self.position)
#         reverse_sir = reverse_signal / interference_to_self if interference_to_self > 0 else float('inf')
#         can_reverse = target.can_deliver_to(self, interference_to_self, threshold, min_sir)
#
#         return {
#             "forward": {
#                 "from": self.id,
#                 "to": target.id,
#                 "signal": forward_signal,
#                 "SIR": forward_sir,
#                 "can_deliver": can_forward
#             },
#             "reverse": {
#                 "from": target.id,
#                 "to": self.id,
#                 "signal": reverse_signal,
#                 "SIR": reverse_sir,
#                 "can_deliver": can_reverse
#             }
#         }
#
#     def __repr__(self):
#         return f"<Node id={self.id} position={self.position} power={self.power}W channel={self.channel}>"
#
#

