import math


class Frequency:
    UNIT_MULTIPLIERS = {
        'Hz': 1,
        'kHz': 1_000,
        'MHz': 1_000_000,
        'GHz': 1_000_000_000
    }

    def __init__(self, value: float, unit: str = 'Hz'):
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Неподдерживаемая единица измерения '{unit}'. "
                             f"Выберите из списка {list(self.UNIT_MULTIPLIERS.keys())}")
        self._hz = float(value) * self.UNIT_MULTIPLIERS[unit]

    @property
    def hz(self):
        return self._hz

    def to(self, unit: str) -> float:
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Unsupported frequency unit '{unit}'. "
                             f"Choose from {list(self.UNIT_MULTIPLIERS.keys())}")
        return self._hz / self.UNIT_MULTIPLIERS[unit]

    def __repr__(self):
        return f"Frequency({self._hz} Hz)"

    def __str__(self):
        # Автоматический выбор наиболее крупной единицы, где значение ≥ 1
        for unit in ['GHz', 'MHz', 'kHz', 'Hz']:
            val = self.to(unit)
            if val >= 1:
                return f"{val:.3f} {unit}"
        return f"{self._hz} Hz"

    def _check_other(self, other):
        # Возвращает количество герц, если other — Frequency или число
        if isinstance(other, Frequency):
            return other.hz
        elif isinstance(other, (int, float)):
            # Интерпретируем как герцы
            return float(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return math.isclose(self._hz, other_hz, rel_tol=1e-9)

    def __lt__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return self._hz < other_hz

    def __le__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return self._hz <= other_hz

    def __gt__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return self._hz > other_hz

    def __ge__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return self._hz >= other_hz

    def __add__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        return Frequency(self._hz + other_hz, 'Hz')

    def __sub__(self, other):
        other_hz = self._check_other(other)
        if other_hz is NotImplemented:
            return NotImplemented
        diff = self._hz - other_hz
        if diff < 0:
            # Если результат отрицательный — возвращаем 0 Hz
            diff = 0.0
        return Frequency(diff, 'Hz')

    def __hash__(self):
        return hash(round(self._hz, 9))


class Watt:
    def __init__(self, value: float):
        if value < 0:
            raise ValueError("Power in watts cannot be negative")
        self._value = float(value)

    @property
    def value(self):
        return self._value

    def to_dbm(self) -> float:
        if self._value == 0:
            return -math.inf
        return 10 * math.log10(self._value * 1000)

    @classmethod
    def from_dbm(cls, dbm_value: float) -> 'Watt':
        return cls(10 ** ((dbm_value - 30) / 10))

    def __repr__(self):
        return f"{self._value} W"

    def __eq__(self, other):
        if isinstance(other, Watt):
            return math.isclose(self._value, other._value, rel_tol=1e-9)
        return NotImplemented

    def __hash__(self):
        return hash(round(self._value, 9))


class Dbm:
    def __init__(self, value: float):
        self._value = float(value)

    @property
    def value(self):
        return self._value

    def to_watt(self) -> float:
        return 10 ** ((self._value - 30) / 10)

    @classmethod
    def from_watt(cls, watt_value: float) -> 'Dbm':
        if watt_value == 0:
            return cls(-math.inf)
        return cls(10 * math.log10(watt_value * 1000))

    def __repr__(self):
        return f"{self._value} dBm"

    def __eq__(self, other):
        if isinstance(other, Dbm):
            return math.isclose(self._value, other._value, rel_tol=1e-9)
        return NotImplemented

    def __hash__(self):
        return hash(round(self._value, 9))


class dB:
    def __init__(self, value: float):
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value

    def to_linear(self) -> float:
        """Перевод из dB в линейное отношение мощности."""
        return 10 ** (self._value / 10)

    @classmethod
    def from_linear(cls, linear_value: float) -> 'dB':
        """Создать dB из линейного отношения мощности."""
        if linear_value <= 0:
            raise ValueError("Линейное значение должно быть > 0 для преобразования в dB")
        return cls(10 * math.log10(linear_value))

    def __repr__(self):
        return f"{self._value:.2f} dB"

    def __eq__(self, other):
        if isinstance(other, dB):
            return math.isclose(self._value, other._value, rel_tol=1e-9)
        return NotImplemented

    # Сложение: линейные значения складываем, переводим обратно в dB
    def __add__(self, other):
        if isinstance(other, dB):
            return dB.from_linear(self.to_linear() + other.to_linear())
        return NotImplemented

    # Вычитание: вычитаем линейные значения, проверяем >0, обратно в dB
    def __sub__(self, other):
        if isinstance(other, dB):
            diff = self.to_linear() - other.to_linear()
            if diff <= 0:
                raise ValueError("Результат вычитания менее или равен нулю, невозможно представить в dB")
            return dB.from_linear(diff)
        return NotImplemented

    # Умножение в dB — это просто сложение значений (log(a*b) = log a + log b)
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return dB(self._value * other)
        elif isinstance(other, dB):
            # Здесь логично либо запретить, либо реализовать умножение линейных, но это редко нужно
            return dB(self._value + other._value)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    # Деление в dB — это вычитание значений (log(a/b) = log a - log b)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Недопустимое деление на 0")
            return dB(self._value - 10 * math.log10(other))
        elif isinstance(other, dB):
            return dB(self._value - other._value)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # other / self в линейных величинах, перевести обратно в dB
            linear_result = other / self.to_linear()
            return dB.from_linear(linear_result)
        return NotImplemented

    # Операторы сравнения по значению
    def __lt__(self, other):
        if isinstance(other, dB):
            return self._value < other._value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, dB):
            return self._value <= other._value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, dB):
            return self._value > other._value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, dB):
            return self._value >= other._value
        return NotImplemented


class Power:
    def __init__(self, value: float, unit: str = 'W'):
        unit = unit.strip().upper()
        if unit == 'W':
            self._watt = Watt(value)
            self._dbm = Dbm(self._watt.to_dbm())
        elif unit == 'DBM':
            self._dbm = Dbm(value)
            watt_value = self._dbm.to_watt()
            if watt_value < 0:
                raise ValueError("Мощность не может быть отрицательной")
            self._watt = Watt(watt_value)
        else:
            raise ValueError("Неверная единица измерения, используется 'W' или 'dBm'")

    @property
    def watt(self) -> Watt:
        return self._watt

    @property
    def dbm(self) -> Dbm:
        return self._dbm

    @classmethod
    def from_watt(cls, value: float):
        return cls(value, 'W')

    @classmethod
    def from_dbm(cls, value: float):
        return cls(value, 'dBm')

    def to_watt(self) -> Watt:
        return self._watt

    def to_dbm(self) -> Dbm:
        return self._dbm

    def ratio_db(self, other: 'Power') -> dB:
        if other._watt.value == 0:
            raise ZeroDivisionError("Division by zero power is not allowed")
        ratio = self._watt.value / other._watt.value
        return dB.from_linear(ratio)

    def __repr__(self):
        return f"Power({self._watt}/{self._dbm})"

    def _check_other(self, other):
        if isinstance(other, Power):
            return other.watt.value
        elif isinstance(other, Watt):
            return other.value
        elif isinstance(other, Dbm):
            return other.to_watt()
        elif isinstance(other, (int, float)):
            return float(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return math.isclose(self._watt.value, other_val, rel_tol=1e-9)

    def __lt__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return self._watt.value < other_val

    def __le__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return self._watt.value <= other_val

    def __gt__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return self._watt.value > other_val

    def __ge__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return self._watt.value >= other_val

    def __add__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        return Power(self._watt.value + other_val, 'W')

    def __sub__(self, other):
        other_val = self._check_other(other)
        if other_val is NotImplemented:
            return NotImplemented
        result = self._watt.value - other_val
        if result < 0:
            raise ValueError("Resulting power cannot be negative")
        return Power(result, 'W')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = self._watt.value * other
            if result < 0:
                raise ValueError("Resulting power cannot be negative")
            return Power(result, 'W')
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Power | dB':
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            result = self._watt.value / other
            if result < 0:
                raise ValueError("Resulting power cannot be negative")
            return Power(result, 'W')
        elif isinstance(other, Power):
            if other._watt.value == 0:
                raise ZeroDivisionError("Division by zero power is not allowed")
            ratio = self._watt.value / other._watt.value
            return dB.from_linear(ratio)
        return NotImplemented

    def __hash__(self):
        return hash(round(self._watt.value, 9))


class DataSize:
    UNIT_MULTIPLIERS = {
        'b': 1,  # бит
        'B': 8,  # байт = 8 бит
        'KB': 8 * 1024,  # килобайт в битах
        'MB': 8 * 1024 ** 2,
        'GB': 8 * 1024 ** 3,
        'TB': 8 * 1024 ** 4,
        'PB': 8 * 1024 ** 5,
    }

    def __init__(self, value: float, unit: str = 'B'):
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Неподдерживаемая единица измерения '{unit}'. "
                             f"Выберите из списка {list(self.UNIT_MULTIPLIERS.keys())}")
        self._bits = float(value) * self.UNIT_MULTIPLIERS[unit]

    @property
    def bits(self):
        return self._bits

    @property
    def B(self):
        return self._bits / 8

    def to(self, unit: str) -> float:
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Неподдерживаемая единица измерения '{unit}'. "
                             f"Выберите из списка {list(self.UNIT_MULTIPLIERS.keys())}")
        return self._bits / self.UNIT_MULTIPLIERS[unit]

    def transfer_time(self, speed: 'DataSpeed'):
        """
        Вычисляет время (в секундах) передачи этого объема данных при заданной скорости передачи.
        speed — объект DataSpeed
        """
        if not isinstance(speed, DataSpeed):
            raise TypeError("Ожидается объект DataSpeed")
        if speed.bps == 0:
            raise ValueError("Скорость передачи не может быть нулевой")
        return self.bits / speed.bps

    def human_readable(self, decimal_places=2, use_iec=True) -> str:
        """
        Возвращает строку с удобочитаемым размером.
        decimal_places — число знаков после запятой
        use_iec — использовать ли IEC приставки (KiB, MiB и т.д.) вместо десятеричных (kB, MB).
        """
        if use_iec:
            # IEC приставки, основанные на 1024
            thresholds = [
                (2 ** 40 * 8, 'TiB'),
                (2 ** 30 * 8, 'GiB'),
                (2 ** 20 * 8, 'MiB'),
                (2 ** 10 * 8, 'KiB'),
                (8, 'B'),
                (1, 'b'),
            ]
        else:
            # Десятеричные приставки
            thresholds = [
                (10 ** 12 * 8, 'TB'),
                (10 ** 9 * 8, 'GB'),
                (10 ** 6 * 8, 'MB'),
                (10 ** 3 * 8, 'kB'),
                (8, 'B'),
                (1, 'b'),
            ]

        bits = self.bits
        for factor, unit in thresholds:
            if bits >= factor:
                value = bits / factor
                return f"{value:.{decimal_places}f} {unit}"
        return f"{bits} b"

    def __repr__(self):
        return f"{self._bits} bits"

    def __str__(self):
        # Автоматический выбор самой крупной единицы, где значение >= 1
        for unit in ['PB', 'TB', 'GB', 'MB', 'KB', 'B', 'b']:
            val = self.to(unit)
            if val >= 1:
                return f"{val:.3f} {unit}"
        # Если меньше 1 бита — выводим просто в битах
        return f"{self._bits} b"

    def _check_other(self, other):
        if isinstance(other, DataSize):
            return other.bits
        elif isinstance(other, (int, float)):
            # Интерпретируем как биты
            return float(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return math.isclose(self._bits, other_bits, rel_tol=1e-9)

    def __lt__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return self._bits < other_bits

    def __le__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return self._bits <= other_bits

    def __gt__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return self._bits > other_bits

    def __ge__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return self._bits >= other_bits

    def __add__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        return DataSize(self._bits + other_bits, 'b')

    def __sub__(self, other):
        other_bits = self._check_other(other)
        if other_bits is NotImplemented:
            return NotImplemented
        diff = self._bits - other_bits
        if diff < 0:
            diff = 0.0
        return DataSize(diff, 'b')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DataSize(self._bits * other, 'b')
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")
            return DataSize(self._bits / other, 'b')
        elif isinstance(other, DataSize):
            # Делим на другой DataSize — возвращаем отношение (безразмерное число)
            return self._bits / other.bits
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")
            return DataSize(self._bits // other, 'b')
        elif isinstance(other, DataSize):
            return self._bits // other.bits
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, DataSize):
            return DataSize(self._bits % other.bits, 'b')
        elif isinstance(other, (int, float)):
            return DataSize(self._bits % other, 'b')
        return NotImplemented

    def __hash__(self):
        return hash(round(self._bits, 9))


class DataSpeed:
    UNIT_MULTIPLIERS = {
        'bps': 1,              # бит в секунду
        'Kbps': 10**3,         # килобит в секунду
        'Mbps': 10**6,         # мегабит в секунду
        'Gbps': 10**9,         # гигабит в секунду
        'Tbps': 10**12,        # терабит в секунду
        'Pbps': 10**15,        # петабит в секунду
    }

    def __init__(self, value: float, unit: str = 'bps'):
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Неподдерживаемая единица измерения '{unit}'. "
                             f"Выберите из списка {list(self.UNIT_MULTIPLIERS.keys())}")
        self._bps = float(value) * self.UNIT_MULTIPLIERS[unit]

    @property
    def bps(self) -> float:
        return self._bps

    @property
    def Bps(self):
        # Байты в секунду
        return self._bps / 8

    def to(self, unit: str) -> float:
        unit = unit.strip()
        if unit not in self.UNIT_MULTIPLIERS:
            raise ValueError(f"Неподдерживаемая единица измерения '{unit}'. "
                             f"Выберите из списка {list(self.UNIT_MULTIPLIERS.keys())}")
        return self._bps / self.UNIT_MULTIPLIERS[unit]

    def data_transferred(self, duration: float) -> DataSize:
        """
        Вычисляет объем данных, который можно передать за duration секунд.
        duration — число (секунды)
        Возвращает DataSize.
        """
        if duration < 0:
            raise ValueError("Время не может быть отрицательным")
        bits = self.bps * duration
        return DataSize(bits, 'b')

    def __repr__(self):
        return f"{self._bps} bps"

    def __str__(self):
        # Автоматический выбор самой крупной единицы, где значение >= 1
        for unit in ['Pbps', 'Tbps', 'Gbps', 'Mbps', 'Kbps', 'bps']:
            val = self.to(unit)
            if val >= 1:
                return f"{val:.3f} {unit}"
        # Если меньше 1 бит/с — выводим просто в битах в секунду
        return f"{self._bps} bps"

    def _check_other(self, other):
        if isinstance(other, DataSpeed):
            return other.bps
        elif isinstance(other, (int, float)):
            # Интерпретируем как бит/с
            return float(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return math.isclose(self._bps, other_bps, rel_tol=1e-9)

    def __lt__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return self._bps < other_bps

    def __le__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return self._bps <= other_bps

    def __gt__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return self._bps > other_bps

    def __ge__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return self._bps >= other_bps

    def __add__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        return DataSpeed(self._bps + other_bps, 'bps')

    def __sub__(self, other):
        other_bps = self._check_other(other)
        if other_bps is NotImplemented:
            return NotImplemented
        diff = self._bps - other_bps
        if diff < 0:
            diff = 0.0
        return DataSpeed(diff, 'bps')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DataSpeed(self._bps * other, 'bps')
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")
            return DataSpeed(self._bps / other, 'bps')
        elif isinstance(other, DataSpeed):
            return self._bps / other.bps
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")
            return DataSpeed(self._bps // other, 'bps')
        elif isinstance(other, DataSpeed):
            return self._bps // other.bps
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, DataSpeed):
            return DataSpeed(self._bps % other.bps, 'bps')
        elif isinstance(other, (int, float)):
            return DataSpeed(self._bps % other, 'bps')
        return NotImplemented

    def __hash__(self):
        return hash(round(self._bps, 9))


