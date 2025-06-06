from connection import InterferenceModel, InterferenceSource, InterferenceTimeFunc
from position import Position
from uav import Node

# Создаём модель интерференции
model = InterferenceModel(noise_floor=1e-9)

model.add_source(InterferenceSource(
    position=Position(50, 50),
    power=5,
    activity=InterferenceTimeFunc.periodic_activity(10, 0.5),
    affected_channels=[1, 2, 3],
    name="Radar"
))

model.add_source(InterferenceSource(
    position=Position(100, 100),
    power=1,
    activity=InterferenceTimeFunc.constant_activity(),
    affected_channels=None,
    name="Jammer"
))

# Создаём ноды
node_A = Node(position=Position(30, 30), power=50, channel=2, name="NodeA", node_id=1)
node_B = Node(position=Position(70, 70), power=40, channel=2, name="NodeB", node_id=2)
node_C = Node(position=Position(120, 120), power=60, channel=2, name="NodeC", node_id=3)

nodes = [node_A, node_B, node_C]

time_sample = 12.5
channel = 2

# Проверка метода measure_link_quality
for sender in nodes:
    for receiver in nodes:
        if sender == receiver:
            continue

        to_target_inter = model.get_total_interference(sender.position, sender.channel, time_sample)
        from_target_inter = model.get_total_interference(receiver.position, receiver.channel, time_sample)

        quality = sender.measure_link_quality(receiver, to_target_inter, from_target_inter)

        print(f"{sender.name} → {receiver.name} | Link Quality (SIR): {quality}")


