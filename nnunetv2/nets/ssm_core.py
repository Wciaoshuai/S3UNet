"""The S3UNet core: shared 2D maps or stage-wise low-rank 3D residuals."""

import math
import torch
from torch import nn
from torch.nn import functional as F

MAP_NAMES = ("B_base", "Wr_u", "Wr_h", "Wz_u", "Wz_h")


def resolve_config(config):
    defaults = dict(
        a_shared=False,
        bias_shared=False,
        rank=16,
        maps={},
        reset_gate=True,
        update_gate=True,
        modulation="none",
        learned_step=False,
        prune_initial=True,
        math_version="unified_v2",
    )
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError("Unsupported S3UNet core configuration")
    result = dict(defaults, **config)
    fixed = dict(
        reset_gate=True,
        update_gate=True,
        modulation="none",
        learned_step=False,
        prune_initial=True,
        math_version="unified_v2",
    )
    if any(result[k] != v for k, v in fixed.items()):
        raise ValueError("This release supports only the fixed S3UNet update")
    if set(result["maps"]) - set(MAP_NAMES):
        raise ValueError("Unknown SSM mapping")
    mappings = {}
    for name in MAP_NAMES:
        value = result["maps"].get(name, "lowrank")
        policy = dict(mode=value, rank=result["rank"]) if isinstance(value, str) else dict(value)
        mode, rank = policy.get("mode"), int(policy.get("rank", result["rank"]))
        if mode not in ("shared", "lowrank") or (mode == "lowrank" and rank not in (8, 16)):
            raise ValueError("Only S1, L16 and L8 mappings are supported")
        mappings[name] = dict(mode=mode, rank=rank if mode == "lowrank" else 0)
    result["maps"] = mappings
    modes = {p["mode"] for p in mappings.values()}
    if modes == {"shared"}:
        if not result["a_shared"] or not result["bias_shared"]:
            raise ValueError("S1 shares decay and biases")
    elif modes == {"lowrank"}:
        if (
            result["a_shared"]
            or result["bias_shared"]
            or len({p["rank"] for p in mappings.values()}) != 1
        ):
            raise ValueError("Low-rank presets use stage-specific decay/bias and one residual rank")
    else:
        raise ValueError("Mixed mapping policies are not part of this model")
    return result


class StageVector(nn.Module):
    def __init__(self, channels, stages, shared, value=0.0):
        super().__init__()
        self.stages, self.shared = tuple(stages), shared
        self.lookup = {s: i for i, s in enumerate(stages)}
        self.value = nn.Parameter(
            torch.full((1 if shared else len(stages), channels), float(value))
        )

    def forward(self, stage):
        return self.value[0 if self.shared else self.lookup[stage]]

    @torch.no_grad()
    def import_values(self, values, decay=False):
        if self.shared:
            selected = values.double()
            average = (
                torch.logsumexp(selected, 0) - math.log(len(selected))
                if decay
                else selected.mean(0)
            )
            self.value.copy_(average.unsqueeze(0))
        else:
            self.value.copy_(values[list(self.stages)])


class StageMap(nn.Module):
    def __init__(self, channels, weight_stages, bias_stages, policy, bias_shared):
        super().__init__()
        self.channels = channels
        self.weight_stages = tuple(weight_stages)
        self.weight_lookup = {s: i for i, s in enumerate(weight_stages)}
        self.mode, self.rank = policy["mode"], policy["rank"]
        if self.rank > channels:
            raise ValueError("Residual rank exceeds the state width")
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mode == "lowrank":
            self.left = nn.Parameter(torch.empty(len(weight_stages), channels, self.rank))
            self.right = nn.Parameter(torch.zeros(len(weight_stages), self.rank, channels))
            nn.init.normal_(self.left, std=1 / math.sqrt(channels))
        self.bias = StageVector(channels, bias_stages, bias_shared)

    def effective_weight(self, stage):
        if self.mode == "shared":
            return self.weight
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            i = self.weight_lookup[stage]
            return self.weight + self.left[i] @ self.right[i]

    def forward(self, x, stage):
        shape = (self.channels, self.channels) + (1,) * (x.ndim - 2)
        conv = F.conv2d if x.ndim == 4 else F.conv3d
        return conv(x, self.effective_weight(stage).reshape(shape), self.bias(stage))

    @torch.no_grad()
    def import_values(self, weights, biases):
        """Preserve the original equal-stage initialization, including live zero residuals."""
        self.bias.import_values(biases)
        base = weights.double().mean(0)
        self.weight.copy_(base)
        if self.mode == "lowrank":
            residual = weights[list(self.weight_stages)].double() - base
            u, sigma, vh = torch.linalg.svd(residual, full_matrices=False)
            root = sigma[:, : self.rank].sqrt()
            left = u[:, :, : self.rank] * root[:, None, :]
            right = root[:, :, None] * vh[:, : self.rank, :]
            zero = root == 0
            generator = torch.Generator(device="cpu").manual_seed(2026 + self.fallback_seed)
            fallback_all = torch.randn(
                len(weights), self.channels, self.rank, generator=generator, dtype=left.dtype
            )
            fallback = fallback_all[list(self.weight_stages)].to(left.device) / math.sqrt(
                self.channels
            )
            self.left.copy_(torch.where(zero[:, None, :], fallback, left))
            self.right.copy_(right)


class S3SSMCore(nn.Module):
    def __init__(self, channels, num_stages, spatial_dim, config):
        super().__init__()
        self.config = resolve_config(config)
        self.channels, self.num_stages, self.spatial_dim = channels, num_stages, spatial_dim
        if spatial_dim not in (2, 3) or num_stages < 2:
            raise ValueError("Expected 2D/3D with at least two scales")
        cfg = self.config
        active = list(range(1, num_stages))
        all_stages = list(range(num_stages))
        self.decay = StageVector(channels, active, cfg["a_shared"])
        self.maps = nn.ModuleDict()
        for name in MAP_NAMES:
            weight_stages = active if name in ("Wr_u", "Wr_h", "Wz_h") else all_stages
            bias_stages = active if name.startswith("Wr_") else all_stages
            self.maps[name] = StageMap(
                channels, weight_stages, bias_stages, cfg["maps"][name], cfg["bias_shared"]
            )
            self.maps[name].fallback_seed = sum((i + 1) * ord(c) for i, c in enumerate(name))
        initial = (torch.arange(1, channels + 1).float() / channels).log()
        with torch.no_grad():
            self.decay.value.copy_(initial.unsqueeze(0).expand_as(self.decay.value))

    def _gate(self, u, h, stage, gate):
        prefix = "Wr" if gate == "r" else "Wz"
        logits = self.maps[prefix + "_u"](u, stage)
        if stage == 0:
            bias = self.maps[prefix + "_h"].bias(stage).to(logits.dtype)
            logits = logits + bias.view(1, self.channels, *([1] * self.spatial_dim))
        else:
            logits = logits + self.maps[prefix + "_h"](h, stage)
        return torch.sigmoid(logits)

    def update_state(self, u, h, stage):
        step = 1 / self.num_stages
        z = self._gate(u, h, stage, "z")
        drive = self.maps["B_base"](u, stage) * z
        if stage == 0:
            beta = torch.ones_like(h) * step
            return F.relu(beta * drive)
        r = self._gate(u, h, stage, "r")
        shape = (1, self.channels) + (1,) * self.spatial_dim
        a = -self.decay(stage).exp().to(h.dtype).view(shape)
        effective_a = a * r
        alpha = torch.exp(step * effective_a)
        beta = torch.ones_like(effective_a) * step
        return F.relu(alpha * h + beta * drive)

    @torch.no_grad()
    def import_dense(self, state):
        self.decay.import_values(state["log_A"], decay=True)
        for name, module in self.maps.items():
            module.import_values(state[name + ".weight"], state[name + ".bias"])
