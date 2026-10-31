from enum import IntEnum


class GameType(IntEnum):
    YON_IKKYOKU = 0  # 4-Player One Kyoku
    YON_TONPUSEN = 1  # 4-Player East (Tonpusen)
    YON_HANCHAN = 2  # 4-Player South (Hanchan)
    SAN_IKKYOKU = 3  # 3-Player One Kyoku
    SAN_TONPUSEN = 4  # 3-Player East
    SAN_HANCHAN = 5  # 3-Player South
