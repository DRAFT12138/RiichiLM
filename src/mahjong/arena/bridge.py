"""Reviewed match profiles shared by the interactive and headless runners."""

from importlib.resources import files
import json

from mahjong.contracts import MahjongError
from mahjong.envs.policy_bridge import PROFILE_DIGEST
from mahjong.inference.context_bridge import ContextDecisionBridge
from mahjong.inference.policy import digest


def profile_for(mode):
    if mode not in ("single", "tonpuu", "hanchan"):
        raise MahjongError("CONFIG_INVALID", "未知赛制")
    return json.loads(files("mahjong.arena").joinpath(
        "resources", "tonpuu.json" if mode == "tonpuu" else "hanchan.json"
    ).read_text())


class ArenaDecisionBridge(ContextDecisionBridge):
    _supported_profiles = frozenset((PROFILE_DIGEST, digest(profile_for("tonpuu"))))
