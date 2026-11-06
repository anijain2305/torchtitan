# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-size Kimi K3 training configs for fake-process-group timing runs.

Each config is the memory study's setting for one parallelism layout: the
Kimi-K3 flavor without the vision encoder, fp32 master weights with bf16
compute, full remat, chunked loss, DistMuon, and a 16M-token global batch.
Run one in fake mode (one process standing in for logical rank 0 of NGPU;
collectives move no data) to time a rank's real compute, non-matmul ops
included:

  NGPU=512 COMM_BACKEND=fake PYTHONPATH=.:scripts/memory_study \\
    MODULE=kimi_k3_fake_configs CONFIG=kimi_k3_fsdp_ep64_8k ./run_train.sh
"""

import importlib.util
import os
import sys
import types
from dataclasses import replace

# The Kimi K3 test recipe (imported below for its DistMuon layout) pulls in the
# multimodal dataloader, which imports torchvision at module level for image
# decoding. These text-only runs never call it, so stand in empty modules when
# torchvision is not installed.
class _Placeholder(types.ModuleType):
    """Any attribute is another placeholder (module-level defaults touch enums)."""

    def __getattr__(self, attr: str) -> "_Placeholder":
        if attr.startswith("__"):  # keep import machinery and pickling away
            raise AttributeError(attr)
        return _Placeholder(f"{self.__name__}.{attr}")


import datasets  # noqa: F401,E402 -- before the placeholders: it probes torchvision at import

_TORCHVISION_PLACEHOLDERS = (
    "torchvision",
    "torchvision.io",
    "torchvision.transforms",
    "torchvision.transforms.v2",
    "torchvision.transforms.v2.functional",
)
_needs_placeholders = importlib.util.find_spec("torchvision") is None
if _needs_placeholders:
    for name in _TORCHVISION_PLACEHOLDERS:
        sys.modules[name] = _Placeholder(name)
    sys.modules["torchvision"].io = sys.modules["torchvision.io"]
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
    sys.modules["torchvision.transforms"].v2 = sys.modules["torchvision.transforms.v2"]
    sys.modules["torchvision.transforms.v2"].functional = sys.modules[
        "torchvision.transforms.v2.functional"
    ]

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer, Optim
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    _KimiK3TrainerConfig,
)

# Only the recipe import needs them; later find_spec("torchvision") checks (e.g.
# Hugging Face datasets) must keep seeing torchvision as absent.
if _needs_placeholders:
    for name in _TORCHVISION_PLACEHOLDERS:
        sys.modules.pop(name, None)

GLOBAL_TOKENS = 16 * 1024 * 1024  # tokens per optimizer step
# MXFP8 for the routed experts only (DistMoE); dense linears stay bf16, as in
# Kimi K3's pre-training. KIMI_K3_DENSE_MXFP8=1 restores MXFP8 dense linears.
DENSE_MXFP8 = os.environ.get("KIMI_K3_DENSE_MXFP8", "0") == "1"


def _balanced_routing(model_config) -> None:
    """Route every MoE layer round-robin with exact balance (TorchTitan's
    RoundRobinTokenChoiceTopKRouter), as the roofline assumes. A freshly
    initialized router routes very unevenly, which a timing run should not
    measure, and which also faults this DistMoE checkout."""
    import dataclasses

    from torchtitan.models.common.moe import RoundRobinTokenChoiceTopKRouter

    names = [f.name for f in dataclasses.fields(RoundRobinTokenChoiceTopKRouter.Config)]
    for layer in model_config.layers:
        if layer.moe is not None:
            old = layer.moe.router
            layer.moe.router = RoundRobinTokenChoiceTopKRouter.Config(
                **{n: getattr(old, n) for n in names if hasattr(old, n)}
            )


def _kimi_k3(
    seq_len: int,
    parallelism: ParallelismConfig,
    steps: int = 4,
    model_transforms=(),
    converters=None,
) -> Trainer.Config:
    """model_transforms rewrite the model config before the trainer config
    validates it (CP needs its attention modules swapped first)."""
    from torchtitan.config.transform import transform_model_config_

    model_config = replace(
        build_model_config(
            "Kimi-K3", seq_len=seq_len, parallelism=parallelism, converters=converters
        ),
        vision_encoder=None,
    )
    if model_transforms:
        model_config = transform_model_config_(model_config, list(model_transforms))
    _balanced_routing(model_config)
    tokens_per_microbatch = seq_len  # one sequence per microbatch
    return _KimiK3TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config)
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config, muon_lr=8e-4, adamw_lr=8e-4, parallelism=parallelism
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_ratio=0.8, decay_type="linear", min_lr_factor=0.0
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
            num_tokens_per_train_step=GLOBAL_TOKENS,
            max_context_length=seq_len,
            steps=steps,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(tokens_per_microbatch),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
        validator=None,
    )


def kimi_k3_fsdp_ep64_8k() -> Trainer.Config:
    """FSDP over 512 GPUs with EP 64, 8K context (4 microbatches per GPU per step)."""
    return _kimi_k3(
        8192,
        ParallelismConfig(data_parallel_shard_degree=-1, expert_parallel_degree=64),
    )


def kimi_k3_fsdp_ep64_32k_cp2() -> Trainer.Config:
    """FSDP over 512 GPUs with EP 64 and CP 2, 32K context."""
    return _kimi_k3(
        32768,
        ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=64,
            context_parallel_degree=2,
        ),
    )


def kimi_k3_fsdp_ep64_8k_distmoe() -> Trainer.Config:
    """FSDP + EP 64 at 8K with the routed experts on DistMoE, as a stand-in
    model: DistMoE needs SwiGLU (not SiTU-GLU) and a power-of-two local expert
    count, so the routed experts use SwiGLU (same GEMMs) and 1,024 experts (16
    per GPU at EP 64; same top-k and expert size, so the same expert FLOPs per
    token, with 12.5% fewer tokens per expert). DistMoE runs in its default
    saved-inputs mode with scratch capacity factor 4, as in the memory study."""
    from unittest import mock

    from torchtitan.config.transform import apply_transforms
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.activation import SwiGLU
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
    from torchtitan.models.kimi_k3 import flavors

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1, expert_parallel_degree=64
    )
    real_config_fn = flavors._kimi_k3_config

    def with_1024_experts(**kwargs):
        return real_config_fn(**{**kwargs, "num_experts": 1024, "vision_encoder": None})

    with mock.patch.object(flavors, "_kimi_k3_config", with_1024_experts):
        config = _kimi_k3(8192, parallelism)
    for layer in config.model.layers:
        if layer.moe is not None:
            layer.moe.routed_experts = replace(
                layer.moe.routed_experts, activation_fn=SwiGLU.Config()
            )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=None, scratch_capacity_factor=4.0
    )
    return apply_transforms(config, [DistMoeTransform()])


def _profiled(config: Trainer.Config, name: str) -> Trainer.Config:
    """Profile the last of the 4 steps (after warmup) into
    outputs/profiling/<name>, for a per-op-class breakdown of one step."""
    from torchtitan.observability.profiler import Profiler

    config.profiler = Profiler.Config(
        enable_profiling=True,
        save_traces_folder=f"profiling/{name}",
        profile_freq=4,
        profiler_warmup=2,
        profiler_active=1,
    )
    return config


def kimi_k3_fsdp_ep64_8k_profiled() -> Trainer.Config:
    """kimi_k3_fsdp_ep64_8k with its last step profiled."""
    return _profiled(kimi_k3_fsdp_ep64_8k(), "fsdp_ep64_8k")


def kimi_k3_fsdp_ep64_8k_distmoe_profiled() -> Trainer.Config:
    """kimi_k3_fsdp_ep64_8k_distmoe with its last step profiled."""
    return _profiled(kimi_k3_fsdp_ep64_8k_distmoe(), "fsdp_ep64_8k_distmoe")


def _long_documents(config: Trainer.Config) -> Trainer.Config:
    """Feed 2,048 synthetic documents of ~50K tokens (outputs/synthetic/long_docs.json; at least one per DP rank),
    so after concat-then-split every 8K window holds one document and varlen
    attention spans the full window -- the roofline's MLA assumption. c4 packs
    many short documents per window, which cuts attention work several-fold."""
    from torchtitan.components.data import (
        HuggingFaceRandomAccessSource,
        SingleDatasetConfig,
    )
    from torchtitan.components.data.text_datasets import TextProcessor

    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": "outputs/synthetic/long_docs.json"},
            ),
            processor=TextProcessor.Config(),
            post_filters=(lambda sample: sample is not None,),
        )
    )
    return config


def kimi_k3_fsdp_ep64_8k_distmoe_longdocs_profiled() -> Trainer.Config:
    """The DistMoE stand-in on full-window documents, last step profiled."""
    return _profiled(
        _long_documents(kimi_k3_fsdp_ep64_8k_distmoe()), "fsdp_ep64_8k_distmoe_longdocs"
    )


def kimi_k3_fsdp_ep64_32k_cp2_distmoe_longdocs_profiled() -> Trainer.Config:
    """Validation config: FSDP + EP 64 with CP 2 at 32K (2 microbatches of 16K
    local tokens per GPU per step), the DistMoE stand-in, full-window documents,
    last step profiled. MLA runs Ulysses CP on varlen (the roofline's CP mode);
    KDA runs its state-passing CP."""
    from unittest import mock

    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.activation import SwiGLU
    from torchtitan.models.common.attention import VarlenInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        UlyssesCPVarlenInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
    from torchtitan.models.kimi_k3 import flavors

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        expert_parallel_degree=64,
        context_parallel_degree=2,
    )
    real_config_fn = flavors._kimi_k3_config

    def with_1024_experts(**kwargs):
        return real_config_fn(**{**kwargs, "num_experts": 1024, "vision_encoder": None})

    cp_transform = ContextParallelTransform(
        inner_attention_map={
            VarlenInnerAttention: UlyssesCPVarlenInnerAttention,
            InnerKDA: ContextParallelInnerKDA,
        }
    )
    with mock.patch.object(flavors, "_kimi_k3_config", with_1024_experts):
        config = _kimi_k3(32768, parallelism, model_transforms=[cp_transform])
    for layer in config.model.layers:
        if layer.moe is not None:
            layer.moe.routed_experts = replace(
                layer.moe.routed_experts, activation_fn=SwiGLU.Config()
            )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=None, scratch_capacity_factor=4.0
    )
    config = apply_transforms(config, [DistMoeTransform()])
    return _profiled(_long_documents(config), "fsdp_ep64_32k_cp2_distmoe_longdocs")


# ---------------------------------------------------------------- memory validation
# Section 4 of the blog: Kimi K3 as the memory study assumes it -- the real 896
# experts, bf16 dense linears (MXFP8 with KIMI_K3_DENSE_MXFP8=1), DistMoE's MXFP8 experts,
# full remat, chunked loss, DistMoE with scratch capacity factor 4, balanced
# routing, full-window documents. SwiGLU stands in for SiTU-GLU (DistMoE
# implements SwiGLU only; same tensor sizes). Each run takes 3 steps and dumps a
# CUDA memory snapshot after step 3 (outputs/memory_snapshot/<name>).


def _patch_dist_moe_1f1b() -> None:
    """DistMoE plans its PP activation slots from a multi-stage schedule
    (``schedule._stages`` and ``pipeline_order``); plain 1F1B is a single-stage
    schedule without them. Give the planner a 1F1B view: warmup of
    num_stages - stage_index forwards (as Schedule1F1B does), then alternating
    backward / forward, then the remaining backwards. Experiment-local shim for
    these memory runs."""
    from torch.distributed.pipelining.schedules import (
        _Action,
        FORWARD,
        FULL_BACKWARD,
        PipelineScheduleSingle,
    )

    from torchtitan.models.common.dist_moe import runtime as dm_runtime

    original = dm_runtime.DistMoeRuntime._plan_pp_activation_slots
    if getattr(original, "_one_f_one_b_shim", False):
        return

    def plan(self, schedule, *, pp_rank, model_parts):
        if not isinstance(schedule, PipelineScheduleSingle):
            return original(self, schedule, pp_rank=pp_rank, model_parts=model_parts)
        stage, m = schedule._stage, schedule._n_microbatches
        s = stage.stage_index
        warmup = min(m, schedule._num_stages - s)
        order = [_Action(s, FORWARD, i) for i in range(warmup)]
        for i in range(m - warmup):
            order += [_Action(s, FULL_BACKWARD, i), _Action(s, FORWARD, warmup + i)]
        order += [_Action(s, FULL_BACKWARD, i) for i in range(m - warmup, m)]
        view = types.SimpleNamespace(
            _stages=[stage], pipeline_order={pp_rank: order}, _n_microbatches=m
        )
        return original(self, view, pp_rank=pp_rank, model_parts=model_parts)

    plan._one_f_one_b_shim = True
    dm_runtime.DistMoeRuntime._plan_pp_activation_slots = plan
    # the runtime also registers a forward context on every local stage
    if not hasattr(PipelineScheduleSingle, "_stages"):
        PipelineScheduleSingle._stages = property(lambda self: [self._stage])


# at import, so it is active in whichever process builds the DistMoE runtime
try:
    _patch_dist_moe_1f1b()
except ImportError:  # dist_moe not installed: the memory-validation configs need it anyway
    pass

_MXFP8_DENSE = [
    # KDA projections, except beta (width -> 96 heads): under TP 2 its 48 rows
    # break MXFP8's 32 x 32 weight tiles
    "delta_attention.q_proj",
    "delta_attention.k_proj",
    "delta_attention.v_proj",
    "delta_attention.output_gate",
    "delta_attention.output_proj",
    "delta_attention.forget_a",
    "delta_attention.forget_b",
    # MLA projections
    "attention.wq_a",
    "attention.wq_b",
    "attention.wkv_a",
    "attention.wkv_b",
    "attention.gate",
    "attention.wo",
    "shared_experts",
    "feed_forward",
    "routed_down",
    "routed_up",
]  # not attention_res_proj (D -> 1)


def _memory_study(
    name: str, seq_len: int, parallelism: ParallelismConfig, *, cp: bool = False
) -> Trainer.Config:
    from torchtitan.config.transform import (
        apply_transforms,
        ContextParallelTransform,
        MXFP8LinearConverter,
    )
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.activation import SwiGLU
    from torchtitan.models.common.attention import VarlenInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        UlyssesCPVarlenInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
    from torchtitan.observability.profiler import Profiler

    transforms = []
    if cp:
        transforms.append(
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: UlyssesCPVarlenInnerAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            )
        )
    config = _kimi_k3(
        seq_len,
        parallelism,
        steps=3,
        model_transforms=transforms,
        converters=(
            [MXFP8LinearConverter.Config(fqns=list(_MXFP8_DENSE))]
            if DENSE_MXFP8
            else None
        ),
    )
    for layer in config.model.layers:
        if layer.moe is not None:
            layer.moe.routed_experts = replace(
                layer.moe.routed_experts, activation_fn=SwiGLU.Config()
            )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=None, scratch_capacity_factor=4.0
    )
    config = apply_transforms(config, [DistMoeTransform(expert_precision="mxfp8")])
    config.profiler = Profiler.Config(
        enable_memory_snapshot=True,
        memory_snapshot_freq=3,
        save_memory_snapshot_folder=f"memory_snapshot/{name}",
    )
    return _long_documents(config)


def _pp(seq_len: int, P: int, tp: int = 1) -> ParallelismConfig:
    dp = 512 // (P * tp)
    return ParallelismConfig(
        pipeline_parallel_degree=P,
        pipeline_parallel_schedule="1F1B",
        num_pp_microbatches=GLOBAL_TOKENS // seq_len // dp,
        data_parallel_shard_degree=-1,
        expert_parallel_degree=512 // P,
        tensor_parallel_degree=tp,
    )


def mem_fsdp_ep64_8k() -> Trainer.Config:
    """FSDP 512, EP 64, CP 1, TP 1 at 8K (blog model: 127 GB)."""
    return _memory_study(
        "fsdp_ep64_8k",
        8192,
        ParallelismConfig(data_parallel_shard_degree=-1, expert_parallel_degree=64),
    )


def mem_fsdp_ep64_32k_cp2() -> Trainer.Config:
    """FSDP 512, EP 64, CP 2 at 32K (blog model: 178 GB)."""
    return _memory_study(
        "fsdp_ep64_32k_cp2",
        32768,
        ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=64,
            context_parallel_degree=2,
        ),
        cp=True,
    )


def mem_pp8_8k() -> Trainer.Config:
    """PP 8, ZeRO-1 (FSDP 64, EP 64 per stage), CP 1, TP 1 at 8K; run with FAKE_PP_RANK=0
    (blog model, first stage: 162 GB)."""
    config = _memory_study("pp8_8k", 8192, _pp(8192, 8))
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    return config


def mem_pp8_cp2_8k() -> Trainer.Config:
    """PP 8, ZeRO-1, CP 2, TP 1 at 8K; run with FAKE_PP_RANK=0 (blog model: 137 GB).
    (TP is not validated: Kimi K3's DistMuon requires tensor_parallel_degree=1.)"""
    parallelism = _pp(8192, 8)
    parallelism.context_parallel_degree = 2
    parallelism.num_pp_microbatches = GLOBAL_TOKENS // 8192 // (512 // 16)
    config = _memory_study("pp8_cp2_8k", 8192, parallelism, cp=True)
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    return config


def mem_pp4_8k() -> Trainer.Config:
    """PP 4, ZeRO-1 (FSDP 128, EP 128 per stage: 7 experts per GPU), CP 1, TP 1 at
    8K, 16 microbatches per pipeline. Run with --comm-backend real_pp_fake_spmd on
    4 GPUs (real PP send/recv, fake SPMD axes); every rank snapshots its stage."""
    from torchtitan.config import CommConfig

    config = _memory_study("pp4_8k", 8192, _pp(8192, 4))
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    # The first step JIT-compiles DistMoE's and the attention residual's kernels;
    # stage 0 can take longer than NCCL's default 300 s while the others wait.
    config.comm = CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800)
    return config
