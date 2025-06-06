from position import Position
import numpy as np
import matplotlib.pyplot as plt
from signal_source import SignalSource, InterferenceSource, Channel
from data_types import Frequency, Power


class InterferenceModel:
    def __init__(self, noise_floor: Power = Power.from_watt(1e-10)):
        """
        :param noise_floor: фоновый уровень шума (Power)
        """
        self.sources: list[InterferenceSource] = []
        self.noise_floor = noise_floor  # Power

    def add_source(self, source: InterferenceSource):
        """Добавить новый источник помех"""
        self.sources.append(source)

    def delete_source(self, source: str | InterferenceSource):
        if isinstance(source, InterferenceSource):
            self.sources.remove(source)
        elif isinstance(source, str):
            for i in self.sources:
                if i.name == source:
                    self.sources.remove(i)
                    break
        else:
            raise ValueError('Неверный тип источника помех')

    def get_total_interference(self, pos: Position, channel: Channel, time: float) -> Power:
        """
        Вычисляет суммарную мощность помех в данной позиции, на частоте (канале), в указанный момент времени.
        Возвращает Power.
        """
        total_power = self.noise_floor  # Power

        for src in self.sources:
            # Проверяем активность и диапазон частот
            if src.is_active(time) and (channel is None or src.is_bandwidth_in_range(channel)):
                # interference_at возвращает Power, с учётом расстояния и path loss
                power_at_pos = src.interference_at(pos, time, channel)
                total_power += power_at_pos

        return total_power

    def plot_heatmap(
            self,
            time: float,
            channel: Channel = None,
            area_size: tuple[int, int] = (200, 200),
            resolution: int = 2,
            log_scale: bool = True,
            title: str = None,
            show_sources: bool = True
    ):
        """
        Рисует карту интерференции.
        :param time: момент времени
        :param channel: частота (канал)
        :param area_size: (ширина, высота) зоны, в метрах
        :param resolution: шаг сетки в метрах
        :param log_scale: логарифмический масштаб мощности
        :param title: заголовок графика
        :param show_sources: отображать ли позиции источников
        """
        width, height = area_size
        x_points = np.arange(0, width, resolution)
        y_points = np.arange(0, height, resolution)

        heatmap = np.zeros((len(y_points), len(x_points)))

        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                pos = Position(x, y)
                p: Power = self.get_total_interference(pos, channel, time)
                heatmap[i, j] = p.to_watt().value

        if log_scale:
            heatmap = np.log10(heatmap + 1e-15)

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, extent=(0, width, 0, height), origin='lower',
                   cmap='hot', interpolation='nearest')
        plt.colorbar(label='log10(Interference Power) [W]')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title or f'Interference Heatmap at t={time:.1f}s, freq={channel}')
        plt.grid(True, color='white', alpha=0.2)

        if show_sources:
            for src in self.sources:
                if src.is_active(time) and (channel is None or src.is_bandwidth_in_range(channel)):
                    plt.plot(src.position.x, src.position.y, 'bo')
                    plt.text(src.position.x + 1, src.position.y + 1, src.name or "Src",
                             color='white', fontsize=8)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<InterferenceModel with {len(self.sources)} sources, noise_floor={self.noise_floor}>"


# Тестирование
def test_interference_model():
    model = InterferenceModel()

    # Источники помех
    src1 = InterferenceSource(
        Position(20, 30),
        Power.from_watt(1e-3),
        freq_range=(Frequency(2.4e9, 'GHz'), Frequency(2.5e9, 'GHz')),  # 2.4-2.5 GHz
        name="Jammer1"
    )
    src2 = InterferenceSource(
        Position(70, 50),
        Power.from_watt(5e-4),
        freq_range=(Frequency(2.45e9, 'GHz'), Frequency(2.55e9, 'GHz')),  # немного другой диапазон
        name="Jammer2"
    )
    src3 = InterferenceSource(
        Position(50, 80),
        Power.from_watt(8e-4),
        freq_range=None,  # широкополосный джаммер
        name="WideJammer"
    )

    model.add_source(src1)
    model.add_source(src2)
    model.add_source(src3)

    model.plot_heatmap(time=0, channel=Channel(Frequency(2.45e9, 'GHz'), Frequency(2.50e9, 'GHz')), area_size=(100, 100), resolution=1,
                       title="Interference Heatmap Test")

if __name__ == "__main__":
    test_interference_model()

