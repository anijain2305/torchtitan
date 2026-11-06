# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchx component that runs a shell script verbatim on whole GB300 hosts.

torchx's built-in ``utils.sh`` shlex-quotes the joined arguments, so a script
with ``;``, pipes or ``$`` reaches bash as a single word and exits 127. This
component passes the script to ``/bin/bash -c`` unchanged.

Usage: see README.md in this directory.
"""

import torchx.specs as specs
from torchx.specs.fb import overlay_mast


def raw_sh(
    script: str,
    num_replicas: int = 1,
    image: str = "conda_mast_core:stable",
    h: str = "nebius_gb300",
    one_scaleup_domain: bool = False,
) -> specs.AppDef:
    """Run ``script`` via ``/bin/bash -c`` on ``num_replicas`` whole hosts.

    ``one_scaleup_domain`` asks MAST to place all hosts inside one NVL72
    scale-up domain (same topology constraint as fb/mast_rl/mast.py).
    """
    role = specs.Role(
        name="sh",
        image=image,
        entrypoint="/bin/bash",
        args=["-c", script],
        num_replicas=num_replicas,
        resource=specs.named_resources[h],
    )
    if one_scaleup_domain:
        overlay_mast.HpcTaskGroupSpec(
            role,
            {
                "topologyConstraints": [
                    {
                        "domain": {
                            "multiple": num_replicas,
                            "quorumBuffer": 0,
                            "maxGroupCount": 1,
                        }
                    }
                ]
            },
        )
    return specs.AppDef(name="raw-sh", roles=[role])
