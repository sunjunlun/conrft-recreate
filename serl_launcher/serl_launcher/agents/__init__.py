from .continuous.bc import BCAgent
from .continuous.sac import SACAgent

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
}

try:
    from .continuous.sac_hybrid_single import SACAgentHybridSingleArm
    agents["sac_hybrid_single"] = SACAgentHybridSingleArm
except ImportError:
    pass

try:
    from .continuous.sac_hybrid_dual import SACAgentHybridDualArm
    agents["sac_hybrid_dual"] = SACAgentHybridDualArm
except ImportError:
    pass