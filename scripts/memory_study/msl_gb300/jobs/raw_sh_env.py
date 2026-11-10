# raw_sh (msl_gb300/raw_sh.py) plus environment variables for the role.
import torchx.specs as specs


def raw_sh_env(
    script: str,
    num_replicas: int = 1,
    image: str = "conda_mast_core:stable",
    h: str = "nebius_gb300",
    env: str = "DISABLE_OILFS=1",
    run_as_root: bool = True,
    one_scaleup_domain: bool = False,
) -> specs.AppDef:
    """Run ``script`` via ``/bin/bash -c`` on ``num_replicas`` whole hosts;
    ``env`` is comma-separated KEY=VALUE pairs. ``one_scaleup_domain`` places all
    hosts in one NVL72 domain (as in raw_sh.py)."""
    import copy

    resource = copy.deepcopy(specs.named_resources[h])
    if run_as_root:  # RDMA verbs device enumeration needs it in the container
        resource.capabilities["MAST_USERNAME"] = "root"
    role = specs.Role(
        name="sh",
        image=image,
        entrypoint="/bin/bash",
        args=["-c", script],
        num_replicas=num_replicas,
        resource=resource,
        env=dict(kv.split("=", 1) for kv in env.split(",") if kv),
    )
    if one_scaleup_domain:
        from torchx.specs.fb import overlay_mast

        overlay_mast.HpcTaskGroupSpec(
            role,
            {"topologyConstraints": [{"domain": {"multiple": num_replicas, "quorumBuffer": 0, "maxGroupCount": 1}}]},
        )
    return specs.AppDef(name="raw-sh", roles=[role])
