from position import Position
from data_types import Power, Frequency, DataSpeed, DataSize, dB
import math
import random
from typing import Callable
from enum import Enum
from scipy.special import erfc


def Q(x):
    """
    Функция Q — хвостовая функция нормального распределения (Q-функция),
    выраженная через дополнительную функцию ошибок (erfc).
    Используется для вычисления вероятности ошибки битов в системах связи.
    :param x: аргумент функции
    :return: вероятность ошибки, соответствующая значению x
    """
    return 0.5 * erfc(x / math.sqrt(2))


class LossExp(Enum):
    free_space = 2.0
    urban = 3.1
    into_forest = 4.0
    underground = 5.0

    @property
    def exp(self):
        return self.value

    @exp.setter
    def exp(self, value):
        raise Exception('Экспонента затухания это эталонное значение')

    def attenuation(self, distance: float) -> float:
        """
        Рассчитывает коэффициент затухания сигнала в зависимости от расстояния.
        :param distance: расстояние от источника сигнала (в метрах)
        :return: коэффициент затухания (distance^экспонента)
        """
        return distance ** self.value

    def __repr__(self):
        return f'<PLE={self.value} ({self.name})>'


class Modulation(Enum):
    BPSK = {
        'bps': 1,
        'M': 2,
        'code_rate': 1 / 2,
        'SNR_threshold': Power(7, 'dBm'),
        'SIR_threshold': Power(10, 'dBm')
    }
    QPSK = {
        'bps': 2,
        'M': 4,
        'code_rate': 1 / 2,
        'SNR_threshold': Power(7, 'dBm'),
        'SIR_threshold': Power(12, 'dBm')
    }
    QAM16 = {
        'bps': 4,
        'M': 16,
        'code_rate': 3/4,
        'SNR_threshold': Power(13, 'dBm'),
        'SIR_threshold': Power(18, 'dBm')
    }
    QAM64 = {
        'bps': 6,
        'M': 64,
        'code_rate': 5 / 6,
        'SNR_threshold': Power(19, 'dBm'),
        'SIR_threshold': Power(24, 'dBm')
    }

    @property
    def bits_per_symbol(self) -> int:
        """
        Количество бит на один символ модуляции.
        Важно для расчёта скорости передачи и энергоэффективности.
        """
        return self.value['bps']

    @property
    def symbols(self) -> int:
        """
        Количество возможных символов в схеме модуляции.
        """
        return self.value['M']

    @property
    def code_rate(self) -> float:
        """
        Коэффициент кодирования — отношение полезных бит к общему числу бит.
        """
        return self.value['code_rate']

    @property
    def SE(self) -> float:
        """
        Спектральная эффективность — количество полезных бит на символ с учётом кодирования.
        """
        return math.log2(self.symbols) * self.code_rate

    @property
    def min_SNR(self) -> Power:
        """
        Минимальное отношение сигнал/шум, необходимое для устойчивой работы.
        """
        return self.value['SNR_threshold']

    @property
    def min_SIR(self) -> Power:
        """
        Минимальное отношение сигнал/помехи, необходимое для устойчивой работы.
        """
        return self.value['SIR_threshold']

    @bits_per_symbol.setter
    def bits_per_symbol(self, value):
        raise Exception('Модуляция это эталонное значение')

    def ber(self, sinr: dB) -> float:
        """
        Рассчитывает вероятность ошибки бита (BER) для данной модуляции и SINR.
        Использует аналитические приближения через функцию Q.
        :param sinr: отношение сигнал/интерференция+шум (в dB)
        :return: вероятность ошибки одного бита
        """
        gamma = sinr.to_linear()
        if self == Modulation.BPSK:
            return Q(math.sqrt(2 * gamma))
        elif self == Modulation.QPSK:
            return Q(math.sqrt(gamma))
        elif self == Modulation.QAM16:
            return 3 / 8 * Q(math.sqrt((4 / 5) * gamma))
        elif self == Modulation.QAM64:
            return 7 / 12 * Q(math.sqrt((6 / 7) * gamma))
        else:
            raise NotImplementedError()

    def per(self, packet_size: int, sinr: dB) -> float:
        """
        Рассчитывает вероятность ошибки пакета (PER) из размера пакета и BER.
        PER = 1 - (1 - BER) ^ packet_size
        :param packet_size: количество бит в пакете
        :param sinr: отношение сигнал/интерференция+шум (в dB)
        :return: вероятность ошибки всего пакета
        """
        ber = self.ber(sinr)
        return 1 - (1 - ber) ** packet_size

    def symbols_rate(self, bandwidth: Frequency, roll_off_factor: float = 0.25):
        """
        Рассчитывает скорость передачи символов (символьную скорость) в Гц.
        Учитывается коэффициент roll-off фильтра.
        :param bandwidth: полоса пропускания (Frequency)
        :param roll_off_factor: коэффициент расширения спектра (обычно 0.2-0.3)
        :return: скорость символов в Гц
        """
        R_s = bandwidth.hz / (1+roll_off_factor)
        return R_s

    def bitrate(self, bandwidth: Frequency) -> DataSpeed:
        """
        Рассчитывает битрейт с учётом спектральной эффективности и полосы.
        :param bandwidth: полоса пропускания (Frequency)
        :return: скорость передачи данных (DataSpeed)
        """
        R_s = self.symbols_rate(bandwidth)
        bitrate = R_s * self.SE
        return DataSpeed(bitrate)

    def throughput(self, bandwidth: Frequency, packet_size: DataSize, sinr: dB) -> DataSpeed:
        """
        Рассчитывает реальную пропускную способность (throughput) с учётом ошибок пакета.
        :param bandwidth: полоса пропускания (Frequency)
        :param packet_size: размер пакета в битах (DataSize)
        :param sinr: отношение сигнал/интерференция+шум (dB)
        :return: реальная скорость передачи данных (DataSpeed)
        """
        bit_rate = self.bitrate(bandwidth).bps
        per = self.per(packet_size.bits, sinr)
        return DataSpeed(bit_rate * (1 - per))

    def __repr__(self):
        return f'<Modulation={self.name} ({self.value} bit per symbol)>'


class Channel:
    def __init__(self,
                 min_freq: Frequency,
                 max_freq: Frequency):
        """
        Инициализация частотного канала с указанием минимальной и максимальной частоты.

        :param min_freq: Минимальная частота канала
        :param max_freq: Максимальная частота канала

        Частоты автоматически переставляются по возрастанию, если пользователь указал их в обратном порядке.
        """
        self.min_freq = min_freq if min_freq <= max_freq else max_freq
        self.max_freq = max_freq if max_freq >= min_freq else min_freq

    @property
    def bandwidth(self) -> Frequency:
        """
        Возвращает ширину полосы частот канала.
        """
        return self.max_freq - self.min_freq

    @property
    def noise_power(self) -> Power:
        """
        Вычисляет тепловой шум на канале по формуле N = k * T * B.
        где k — постоянная Больцмана ≈ 1.38e-23, T — температура, B — ширина полосы в Гц.
        Использует физическую температуру 290 K (около комнатной).
        :return: мощность шума (Power)
        """

        temperature: float = 290
        k = 1.38064852e-23  # Дж/К
        bandwidth_Hz = self.bandwidth.hz
        noise_watt = k * temperature * bandwidth_Hz
        return Power.from_watt(noise_watt)

    def set_channel(self, min_freq: Frequency, max_freq: Frequency):
        """
        Устанавливает границы канала, гарантируя корректный порядок частот.
        :param min_freq: нижняя граница частоты
        :param max_freq: верхняя граница частоты
        """
        self.min_freq = min_freq if min_freq <= max_freq else max_freq
        self.max_freq = max_freq if max_freq >= min_freq else min_freq

    def get_tuple(self):
        """
        Возвращает кортеж (min_freq, max_freq) для удобства передачи данных.
        """
        return (self.min_freq, self.max_freq)

    @classmethod
    def from_tuple(cls, freqs: tuple[Frequency, Frequency]):
        """
        Создаёт объект Channel из кортежа частот.
        """
        return cls(freqs[0], freqs[1])

    def __repr__(self):
        return f'<Channel {self.min_freq.__str__()} - {self.max_freq.__str__()}>'


class SignalSource:
    def __init__(
        self,
        position: Position,
        max_power: Power,
        freq_range: tuple[Frequency, Frequency] | Channel | None,
        path_loss_exponent: LossExp = LossExp.free_space,
        power_factor: float = 1
    ):
        """
        Инициализация базового источника сигнала.

        :param position: Координаты источника сигнала (объект Position)
        :param max_power: Максимальная мощность передатчика (объект Power)
        :param freq_range: Диапазон частот, на которых может работать источник.
                           Может быть кортежем (min_freq, max_freq), объектом Channel или None (широкополосный)
        :param path_loss_exponent: Коэффициент затухания сигнала с расстоянием (обычно >= 2.0, по умолчанию - свободное пространство)
        :param power_factor: Множитель мощности (например, для регулировки эффективной мощности), по умолчанию 1.0

        Атрибут freq_range преобразуется к кортежу, если передан объект Channel.
        """
        self.position: Position = position
        self._max_power: Power = max_power
        self._power_factor: float = power_factor
        self.freq_range: tuple[Frequency, Frequency] | None = freq_range if not isinstance(freq_range, Channel) else freq_range.get_tuple()
        self._path_loss_exponent: LossExp = path_loss_exponent

    @property
    def path_loss_exponent(self):
        """
        Параметр затухания сигнала (экспонента потерь в модели затухания).
        Обычно коэффициент ≥ 2.0 (например, 2 для свободного пространства).
        """
        return self._path_loss_exponent

    @path_loss_exponent.setter
    def path_loss_exponent(self, value: LossExp):
        """
        Устанавливает коэффициент затухания.
        :param value: новый параметр класса LossExp
        """
        self._path_loss_exponent = value

    @property
    def power(self):
        """
        Текущая выходная мощность источника с учётом процента работы.
        Возвращает Power-объект.
        """
        return self._max_power * self._power_factor

    @property
    def max_power(self):
        """
        Максимальная мощность источника (Power-объект).
        """
        return self._max_power

    @max_power.setter
    def max_power(self, value: Power):
        """
        Устанавливает максимальную мощность источника.
        :param value: Power — новая максимальная мощность
        """
        self._max_power = value

    @property
    def power_factor(self):
        """
        Процент работы передатчика в диапазоне [0,1], который влияет на выходную мощность.
        """
        return self._power_factor

    @power_factor.setter
    def power_factor(self, value: float | int):
        """
        Устанавливает процент работы передатчика.
        Если передано целое число, интерпретируется как процент (например, 50 → 0.5).
        Отрицательные значения игнорируются.
        :param value: float или int — новый множитель мощности
        """
        if value < 0:
            return

        if isinstance(value, int):
            self._power_factor = value/100
        elif isinstance(value, float):
            self._power_factor = value

    def signal_at(self, pos: Position) -> Power:
        """
        Вычисляет мощность сигнала в точке `pos` с учётом расстояния и модели затухания.
        Если позиция совпадает с источником (расстояние 0), возвращается полная мощность.
        :param pos: координаты точки приёма
        :return: мощность сигнала (Power)
        """
        distance = self.position.distance_to(pos)  # должен возвращать float в метрах, например
        if distance == 0:
            return self.power
        attenuation = self.path_loss_exponent.attenuation(distance)
        received_power_watt = self.power.watt.value / attenuation
        return Power.from_watt(received_power_watt)

    def SNR_at(self, pos: Position, noise_power: Power = Power(1e-9, 'W')) -> dB:
        """
        Вычисляет отношение сигнал/шум (SNR, Signal-to-Noise Ratio) в точке `pos`.
        :param pos: координаты точки приёма
        :param noise_power: мощность шума (по умолчанию 1 нВт)
        :return: SNR в dB
        """
        signal = self.signal_at(pos)
        SNR = signal / noise_power
        return SNR

    def SINR_at(self, pos: Position, interference_power: Power, noise_power: Power = Power(1e-9, 'W')) -> dB:
        """
        Вычисляет отношение сигнал/(интерференция + шум) (SINR) в точке `pos`.
        :param pos: координаты точки приёма
        :param interference_power: мощность помех
        :param noise_power: мощность шума
        :return: SINR в dB
        """
        signal = self.signal_at(pos)
        total_interference = interference_power + noise_power
        SINR = signal / total_interference
        return SINR

    def SIR_at(self, pos: Position, interference_power_at_pos: Power = Power(0, 'W')) -> dB:
        """
        Вычисляет отношение сигнал/помехи (SIR) без учёта шума.
        :param pos: координаты точки приёма
        :param interference_power_at_pos: мощность помех
        :return: SIR в dB
        """
        signal_power = self.signal_at(pos)

        if interference_power_at_pos.watt.value == 0:
            # Нет помех
            return dB(float('inf'))

        SIR = signal_power / interference_power_at_pos
        return SIR

    def is_frequency_in_range(self, freq: Frequency) -> bool:
        """
        Проверяет, находится ли заданная частота внутри допустимого диапазона источника.
        Если диапазон не ограничен (None), всегда возвращает True.
        :param freq: частота для проверки
        :return: True, если частота в диапазоне
        """

        if self.freq_range is None:
            return True

        return self.freq_range[0] <= freq <= self.freq_range[1]

    def is_bandwidth_in_range(self, freqs: tuple[Frequency, Frequency] | Channel) -> bool:
        """
        Проверяет, полностью ли заданный диапазон частот лежит внутри freq_range источника.
        :param freqs: диапазон частот или Channel
        :return: True, если весь диапазон поддерживается источником
        """

        if self.freq_range is None:
            return True

        lower, upper = freqs if not isinstance(freqs, Channel) else freqs.get_tuple()
        range_lower, range_upper = self.freq_range

        # Проверяем, что нижняя и верхняя частоты проверяемого диапазона лежат внутри self.freq_range
        return range_lower <= lower <= range_upper and range_lower <= upper <= range_upper

    def get_supported_channel(self, bandwidth: Frequency) -> Channel | None:
        """
        Случайным образом выбирает канал заданной ширины в пределах freq_range.
        Возвращает None, если ширина превышает доступный диапазон.
        :param bandwidth: ширина канала (Frequency)
        :return: Channel или None
        """
        if self.freq_range is None:
            return None

        min_f, max_f = self.freq_range
        available = max_f - min_f

        if bandwidth > available:
            return None

        start = Frequency(random.uniform(min_f.hz, max_f.hz - bandwidth.hz))
        end = start + bandwidth
        return Channel(start, end)

    def max_reach_distance(self, min_power: Power) -> float:
        """
        Максимальное расстояние, на котором мощность сигнала будет не ниже min_power.
        Рассчитывается из формулы затухания мощности.
        :param min_power: минимальная необходимая мощность сигнала
        :return: максимальное расстояние в метрах (float)
        """
        if self.power.watt <= min_power.watt:
            return 0.0
        return (self.power.watt / min_power.watt) ** (1 / self.path_loss_exponent.value)

    def move(self, x: float = 0, y: float = 0, z: float = 0):
        """
        Смещает позицию источника сигнала на указанные смещения по осям.
        :param x: смещение по X
        :param y: смещение по Y
        :param z: смещение по Z
        """
        self.position.change_xyz(x, y, z)

    def move_to(self, x: float = None, y: float = None, z: float = None):
        """
        Перемещает источник в абсолютные координаты (если параметр не задан, координата не изменяется).
        :param x: новая координата X (или None — без изменений)
        :param y: новая координата Y (или None)
        :param z: новая координата Z (или None)
        """
        dx = x - self.position.x if x else 0
        dy = y - self.position.y if y else 0
        dz = z - self.position.z if z else 0

        self.position.change_xyz(dx, dy, dz)

    def set_pos(self, x: float = 0, y: float = 0, z: float = 0):
        """
        Устанавливает позицию источника в абсолютных координатах.
        :param x: координата X
        :param y: координата Y
        :param z: координата Z
        """
        self.position.set_xyz(x, y, z)

    def __repr__(self):
        return (f"<SignalSource pos={self.position} power={self.power} ({self._power_factor*100}% {self.max_power}), freq_range={self.freq_range}"
                f"path_loss_exp={self.path_loss_exponent}>")


class Transmitter(SignalSource):
    def __init__(
        self,
        position: Position,
        max_power: Power,
        freq_range: tuple[Frequency, Frequency] | Channel,
        current_channel: tuple[Frequency, Frequency] | Channel,
        path_loss_exponent: LossExp = LossExp.free_space,
        modulation: Modulation = Modulation.BPSK,
    ):
        """
        Инициализация передатчика — источника сигнала с возможностью выбора текущего рабочего канала и модуляции.

        :param position: Положение передатчика (объект Position)
        :param max_power: Максимальная выходная мощность передатчика (Power)
        :param freq_range: Допустимый диапазон частот (кортеж или Channel), в котором передатчик может работать
        :param current_channel: Текущий рабочий канал (кортеж или Channel), должен быть внутри freq_range
        :param path_loss_exponent: Коэффициент затухания сигнала (float), по умолчанию — свободное пространство
        :param modulation: Тип модуляции сигнала (enum Modulation), по умолчанию BPSK

        При инициализации проверяется, что текущий канал входит в допустимый диапазон частот передатчика.
        Мощность передатчика корректируется через power_factor = 0.5 (половина максимальной мощности).
        """
        super().__init__(position, max_power, freq_range, path_loss_exponent, power_factor=0.5)
        if not self.is_bandwidth_in_range(current_channel):
            raise ValueError('Указанная частота канала не входит в диапазон передатчика')
        self._channel: Channel = current_channel
        self.modulation: Modulation = modulation

    @property
    def min_SIR(self):
        """
        Минимально необходимое отношение сигнал/помехи (SIR) для текущей модуляции.
        Используется для оценки качества канала.
        """
        return self.modulation.min_SIR

    @property
    def min_SNR(self):
        """
        Минимально необходимое отношение сигнал/шум (SNR) для текущей модуляции.
        """
        return self.modulation.min_SNR

    @property
    def channel(self) -> Channel:
        """
        Текущий рабочий канал передатчика (Channel).
        """
        return self._channel

    @channel.setter
    def channel(self, freqs: Channel | tuple[Frequency, Frequency]):
        """
        Устанавливает рабочий канал передатчика.
        Проверяет, что канал лежит в пределах допустимого диапазона freq_range.
        :param freqs: новый канал (Channel или кортеж частот)
        :raises ValueError: если канал вне диапазона
        """
        if self.is_bandwidth_in_range(freqs):
            self._channel = freqs if isinstance(freqs, Channel) else Channel.from_tuple(freqs)
        else:
            raise ValueError(f"Частоты {freqs} вне диапазона {self.freq_range}")

    @property
    def channel_capacity(self) -> DataSpeed:
        """
        Максимальная пропускная способность канала (DataSpeed),
        вычисляется как ширина канала (в Гц) × бит на символ модуляции.
        """
        bandwidth_hz = self.channel.bandwidth.hz
        capacity_bps = bandwidth_hz * self.modulation.bits_per_symbol
        return DataSpeed(capacity_bps)

    def SNR_at(self, pos: Position, noise_power: Power = None) -> dB:
        """
        Рассчитывает отношение сигнал/шум (SNR) в позиции с учётом шумовой мощности канала по умолчанию.
        :param pos: позиция приёма
        :param noise_power: мощность шума, если None — берётся из текущего канала
        :return: SNR в dB
        """
        result = super().SNR_at(pos, self.channel.noise_power if noise_power is None else noise_power)
        return result

    def SINR_at(self, pos: Position, interference_power: Power, noise_power: Power = None) -> dB:
        """
        Рассчитывает отношение сигнал/(интерференция + шум) в позиции.
        Если шум не задан — используется шумовая мощность текущего канала.
        :param pos: позиция приёма
        :param interference_power: мощность помех
        :param noise_power: мощность шума (по умолчанию из канала)
        :return: SINR в dB
        """
        result = super().SINR_at(pos, interference_power, self.channel.noise_power if noise_power is None else noise_power)
        return result

    def set_modulation(self, modulation: Modulation = Modulation.BPSK):
        """
        Установить используемую модуляцию для передатчика.
        :param modulation: объект модуляции (по умолчанию BPSK)
        """
        self.modulation = modulation

    def calculate_per(self, packet_size_bits: int, sinr: dB) -> float:
        """
        Рассчитать вероятность ошибки пакета (PER) для текущей модуляции
        при заданном значении SINR и размере пакета.
        :param packet_size_bits: размер пакета в битах
        :param sinr: отношение сигнал/шум+помеха (в дБ)
        :return: вероятность ошибки всего пакета (PER)
        """
        return self.modulation.per(packet_size_bits, sinr)

    def can_transmit(self, data_speed: DataSpeed) -> bool:
        """
        Проверить, поддерживает ли текущая модуляция и полоса пропускания
        передачу с заданной скоростью данных.

        :param data_speed: требуемая скорость передачи данных
        :return: True, если достижимо, иначе False
        """
        bandwidth = self.channel.bandwidth
        bps = self.modulation.bits_per_symbol
        achievable_speed = DataSpeed(bps * bandwidth.hz)
        return achievable_speed >= data_speed

    def probability_of_transfer(self, pos: Position, interference_power: Power, package_length: DataSize) -> float:
        """
        Вычислить вероятность успешной передачи пакета с учетом
        SINR в заданной позиции и мощности помех.
        Используется BER, вычисляемый текущей модуляцией, и размер пакета.
        :param pos: позиция приёмника
        :param interference_power: мощность помех
        :param package_length: размер пакета в битах
        :return: вероятность успешной передачи пакета
        """
        sinr = self.SINR_at(pos, interference_power)
        ber = self.modulation.ber(sinr)
        p_succ = (1 - ber)**package_length.bits
        return p_succ

    def transfer_expected_attempts(self, pos: Position, interference_power: Power, package_length: DataSize) -> int:
        """
        Оценить ожидаемое число попыток передачи пакета
        до успешной передачи (с учётом вероятности успеха).

        :param pos: позиция приёмника
        :param interference_power: мощность помех
        :param package_length: размер пакета
        :return: ожидаемое число попыток (целое число)
        """
        p_succ = self.probability_of_transfer(pos, interference_power, package_length)
        e_n = round(1/p_succ)
        return int(e_n)

    def processing_delay(self):
        """
        Случайная задержка обработки пакета перед передачей,
        моделирующая системные задержки (например, в железе).

        :return: задержка в секундах
        """
        return random.uniform(0.001, 0.01)

    def queuing_delay(self):
        """
        Случайная задержка в очереди перед началом передачи,
        моделирующая нагрузку и ожидание.

        :return: задержка в секундах
        """
        return random.uniform(0.005, 0.02)

    def transmission_delay(self, package_size: DataSize, pos: Position, interference_power: Power) -> float:
        """
        Вычислить время передачи пакета с учётом
        текущей пропускной способности канала и условий среды.

        :param package_size: размер пакета данных
        :param pos: позиция приёмника
        :param interference_power: мощность помех
        :return: время передачи в секундах
        """
        sinr = self.SINR_at(pos, interference_power)
        throughput = self.modulation.throughput(self.channel.bandwidth, package_size, sinr)
        return package_size / throughput

    def attempt_send_latency(self, package_size: DataSize, pos: Position, interference_power: Power) -> float:
        """
        Рассчитать задержку одной попытки отправки пакета,
        включая обработку, очередь и передачу.

        :param package_size: размер пакета
        :param pos: позиция приёмника
        :param interference_power: мощность помех
        :return: суммарная задержка попытки в секундах
        """
        processing = self.processing_delay()
        queuing = self.queuing_delay()
        transmission = self.transmission_delay(package_size, pos, interference_power)
        return processing + queuing + transmission

    def send_latency(self, package_size: DataSize, package_length: DataSize, pos: Position, interference_power: Power) -> float:
        """
        Рассчитать ожидаемую суммарную задержку передачи пакета
        с учётом необходимого числа попыток (из-за ошибок).

        :param package_size: размер пакета для передачи
        :param package_length: длина пакета для расчёта ошибок
        :param pos: позиция приёмника
        :param interference_power: мощность помех
        :return: ожидаемая задержка передачи пакета в секундах
        """
        expected_attempts = self.transfer_expected_attempts(pos, interference_power, package_length)
        attempt_send_latency = self.attempt_send_latency(package_size, pos, interference_power)
        return expected_attempts * attempt_send_latency

    def __repr__(self):
        return f'<Transmitter: POS={self.position}, channel={self.channel} ({self.freq_range}), modulation={self.modulation.name}>'


class InterferenceTimeFunc:
    @staticmethod
    def constant_activity() -> Callable[[float], bool]:
        """
        Возвращает функцию активности помехи, которая всегда активна.

        :return: функция, принимающая время t и всегда возвращающая True
        """
        return lambda t: True

    @staticmethod
    def impulse_activity(start: float, end: float) -> Callable[[float], bool]:
        """
        Возвращает функцию активности помехи с импульсным характером:
        помеха активна только в интервале времени [start, end].

        :param start: время начала активности
        :param end: время окончания активности
        :return: функция, принимающая время t и возвращающая True, если t в интервале
        """
        return lambda t: start <= t <= end

    @staticmethod
    def periodic_activity(period: float, duty: float = 0.5) -> Callable[[float], bool]:
        """
        Возвращает функцию активности помехи с периодическим характером.
        Помеха активна в течение доли duty каждого периода.

        :param period: длительность одного периода активности и неактивности
        :param duty: доля периода, в течение которой помеха активна (0 < duty <= 1)
        :return: функция, принимающая время t и возвращающая True, если в активной фазе
        """
        def f(t: float) -> bool:
            phase = t % period
            return phase < period * duty

        return f

    @staticmethod
    def random_activity(probability: float) -> Callable[[float], bool]:
        """
        Возвращает функцию активности помехи с вероятностным характером.
        В каждый момент времени помеха активна с заданной вероятностью.

        :param probability: вероятность активности помехи (0 <= probability <= 1)
        :return: функция, принимающая время t и возвращающая True с вероятностью probability
        """
        return lambda t: random.random() < probability


class InterferenceSource(SignalSource):
    def __init__(
        self,
        position: Position,
        max_power: Power,
        activity: Callable[[float], bool] = InterferenceTimeFunc.constant_activity(),
        freq_range: tuple[Frequency, Frequency] | None = None,
        path_loss_exponent: LossExp = LossExp.free_space,
        name: str = None
    ):
        """
        Инициализация источника помех.

        :param position: положение источника помех
        :param max_power: максимальная мощность помех в виде объекта Power
        :param activity: функция активности помехи — принимает время (float) и возвращает bool,
                         определяет, активна ли помеха в данный момент
        :param freq_range: частотный диапазон работы источника (нижняя и верхняя частоты),
                           None — если источник широкополосный
        :param path_loss_exponent: коэффициент затухания сигнала (экспонента)
        :param name: необязательное имя источника помех
        """
        super().__init__(position, max_power, freq_range, path_loss_exponent)
        self.activity = activity
        self.name = name

    def is_active(self, time: float) -> bool:
        """
        Проверяет, активен ли источник помехи в заданный момент времени.

        :param time: момент времени (в секундах)
        :return: True, если помеха активна, иначе False
        """
        return self.activity(time)

    def interference_at(self, pos: Position, time: float = 0, channel: Channel = None) -> Power:
        """
        Рассчитывает мощность помехи в точке pos в момент времени time на частоте channel.

        Если помеха неактивна в этот момент времени или частота вне диапазона действия,
        возвращается мощность 0 ватт.

        :param pos: позиция, в которой измеряется мощность помехи
        :param time: момент времени
        :param channel: частота канала (если None, проверка частоты не производится)
        :return: мощность помехи в данной точке и момент времени в виде объекта Power
        """
        if not self.is_active(time):
            return Power.from_watt(0.0)
        if channel is not None and not self.is_bandwidth_in_range(channel):
            return Power.from_watt(0.0)
        return self.signal_at(pos)

    def __repr__(self):
        return f"<InterferenceSource '{self.name}' power={self.power} channels={self.freq_range}>"