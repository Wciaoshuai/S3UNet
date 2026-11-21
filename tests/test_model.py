"""Self-contained CPU contract tests. No datasets or weight fixtures are required."""

import copy
import pytest
import torch
from s3unet_v2 import build_model, load_weights
from s3unet_v2.initialization import parameter_counts

torch.set_num_threads(2)


@pytest.mark.parametrize(
    "dataset,total,core",
    [
        (705, 14211146, 3024),
        (704, 23819368, 20864),
        (703, 23679123, 3024),
        (702, 14296990, 163184),
        (710, 14066259, 14040),
    ],
)
def test_fixed_presets(dataset, total, core):
    model = build_model(dataset)
    counts = parameter_counts(model)
    assert counts["total"] == total and counts["core"] == core
    assert counts["heads"] == sum(p.numel() for p in model.decoder.seg_layers[-1].parameters())
    with pytest.raises(ValueError):
        build_model(dataset, variant="U")


def test_strict_checkpoint_loading():
    training = build_model(705, deep_supervision=True)
    model = build_model(705)
    report = load_weights(
        model, {"network_weights": {"model." + k: v for k, v in training.state_dict().items()}}
    )
    assert len(report["ignored_auxiliary_keys"]) == 12
    state = copy.copy(training.state_dict())
    state["unrelated.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        load_weights(model, state)
    state = copy.copy(model.state_dict())
    state.pop("decoder.ablation_core.maps.B_base.weight")
    with pytest.raises(RuntimeError):
        load_weights(model, state)


@pytest.mark.parametrize("dataset,channels,shape", [(705, 3, (64, 128)), (710, 1, (32, 64, 64))])
def test_forward_checkpointing_and_gradients(dataset, channels, shape):
    model = build_model(dataset, deep_supervision=True).train()
    x = torch.randn(1, channels, *shape)
    state = copy.deepcopy(model.state_dict())
    outputs = model(x)
    loss = sum(y.square().mean() for y in outputs)
    loss.backward()
    expected = [y.detach().clone() for y in outputs]
    gradients = {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    model.load_state_dict(state)
    model.set_gradient_checkpointing(True, True)
    actual = model(x)
    sum(y.square().mean() for y in actual).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    for name, p in model.named_parameters():
        if name in gradients:
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-5, rtol=1e-4)
