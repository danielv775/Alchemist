
from enum import Enum

ADJUSTED_CLOSE = 'AdjClose'
CLOSE = 'Close'
DATE = 'Date'
HIGH = 'High'
LOW = 'Low'
OPEN = 'Open'
VOLUME = 'Volume'
FEATURES = 'Features'
SYMBOLS = 'Symbols'
BUY = 'Buy'
HOLD = 'Hold'
SELL = 'Sell'
SIGNAL = 'Signal'
TARGET = 'Target'
CASH = 'CASH'
DELTA_CASH = 'DeltaCash'
DELTA_HOLDING = 'DeltaHolding'
PORT_VALUE = 'PortValue'


class Phase(Enum):
    TRAINING = 'Training Phase'
    TEST = 'Test Phase'
    VALIDATION = 'Validation Phase'
    # LIVE = 'Live Phase'
