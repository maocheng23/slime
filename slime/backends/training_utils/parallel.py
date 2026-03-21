from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


@dataclass
class ParallelState:
    dp_rank: int
    dp_src_rank: int
    dp_size: int
    cp_rank: int
    cp_size: int
    cp_comm_type: Optional[str]
    dp_cp_rank: int
    dp_cp_size: int
    dp_group: dist.ProcessGroup | None
    dp_cp_group: dist.ProcessGroup | None
    dp_cp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    dp_mesh: dist.DeviceMesh | None
    cp_mesh: dist.DeviceMesh | None
    is_pp_last_stage: bool
    vpp_size: int | None
    microbatch_group_size_per_vp_stage: int | None

    def __init__(self):
        self.vpp_size = 1
        self.microbatch_group_size_per_vp_stage = None
        self.is_pp_last_stage = True
        self.cp_comm_type = None

    @property
    def is_ulysses_cp(self) -> bool:
        """True when using Ulysses (a2a) context parallelism."""
        return self.cp_size > 1 and self.cp_comm_type == "a2a"
