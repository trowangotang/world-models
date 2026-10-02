import numpy as np
import pytest
import torch

from worldmodels.data import collect_rollouts, episode_paths
from worldmodels.env import GridDodgeEnv
from worldmodels.vae.dataset import iterate_minibatches, load_split, split_episodes, to_tensor
from worldmodels.vae.loss import kl_divergence, object_weight_map, vae_loss
from worldmodels.vae.model import ConvVAE, VAEConfig


@pytest.fixture(scope="module")
def frames() -> np.ndarray:
    env = GridDodgeEnv(seed=0)
    return np.stack([env.reset(seed=s) for s in range(16)])


def test_model_shapes():
    model = ConvVAE(VAEConfig(latent_dim=8))
    x = torch.rand(5, 3, 64, 64)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (5, 8)
    assert recon.min() >= 0 and recon.max() <= 1


def test_eval_mode_is_deterministic():
    model = ConvVAE().eval()
    x = torch.rand(2, 3, 64, 64)
    assert torch.equal(model(x)[0], model(x)[0])


def test_save_and_load_roundtrip(tmp_path):
    model = ConvVAE(VAEConfig(latent_dim=4)).eval()
    path = tmp_path / "vae.pt"
    model.save(path, note="hei")
    loaded, ckpt = ConvVAE.load(path)
    x = torch.rand(2, 3, 64, 64)
    assert loaded.config.latent_dim == 4
    assert ckpt["note"] == "hei"
    assert torch.allclose(model(x)[0], loaded(x)[0])


def test_object_weight_map_marks_non_background(frames):
    x = to_tensor(frames[:1])
    w = object_weight_map(x, object_weight=10.0)
    assert w.shape == (1, 1, 64, 64)
    assert set(torch.unique(w).tolist()) == {1.0, 10.0}
    # 6 hindringer + mål = 7 hele celler, pluss agenten på 6x6 piksler
    assert int((w == 10.0).sum()) == 7 * 64 + 36


def test_kl_is_zero_for_standard_normal():
    mu, logvar = torch.zeros(3, 4), torch.zeros(3, 4)
    assert torch.allclose(kl_divergence(mu, logvar), torch.zeros(3))
    assert (kl_divergence(torch.ones(3, 4), logvar) > 0).all()


def test_object_weight_increases_penalty_for_missing_objects(frames):
    x = to_tensor(frames[:4])
    background_only = torch.ones_like(x) * x[:, :, :1, :1]  # hele bildet i bakgrunnsfarge
    mu, logvar = torch.zeros(4, 2), torch.zeros(4, 2)
    plain = vae_loss(background_only, x, mu, logvar, object_weight=1.0)["recon"]
    weighted = vae_loss(background_only, x, mu, logvar, object_weight=10.0)["recon"]
    assert weighted == pytest.approx(10 * plain.item(), rel=1e-4)


def test_model_can_overfit_small_batch(frames):
    torch.manual_seed(0)
    model = ConvVAE(VAEConfig(latent_dim=8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = to_tensor(frames[:8])
    first = None
    for _ in range(60):
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.7 * first


def test_split_is_by_episode_and_disjoint(tmp_path):
    collect_rollouts(tmp_path, num_episodes=20, seed=0)
    train, val = split_episodes(episode_paths(tmp_path), val_fraction=0.2, seed=0)
    assert len(val) == 4 and len(train) == 16
    assert not set(train) & set(val)
    split = load_split(tmp_path, val_fraction=0.2)
    assert split.train.dtype == np.uint8 and split.train.shape[1:] == (64, 64, 3)


def test_iterate_minibatches_covers_all_frames(frames):
    batches = list(iterate_minibatches(frames, batch_size=5, shuffle=True, rng=np.random.default_rng(0)))
    assert [b.shape[0] for b in batches] == [5, 5, 5, 1]
    assert batches[0].dtype == torch.float32 and batches[0].max() <= 1.0


def test_grid_metrics_perfect_and_missing_agent(frames):
    from worldmodels.vae.evaluate import grid_metrics

    perfect = grid_metrics(frames, frames)
    assert perfect == {"cell_accuracy": 1.0, "layout_exact": 1.0, "agent_cell_accuracy": 1.0, "agent_missing": 0.0}
    blank = np.broadcast_to(frames[:, :1, :1], frames.shape).copy()  # bare bakgrunn
    m = grid_metrics(frames, blank)
    assert m["agent_missing"] == 1.0 and m["agent_cell_accuracy"] == 0.0
    assert 0.8 < m["cell_accuracy"] < 1.0


def test_probe_learns_linearly_separable_labels():
    from worldmodels.vae.evaluate import train_probe

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    z = np.eye(4)[y] * 3 + rng.normal(scale=0.3, size=(400, 4))
    acc = train_probe(z[:300], y[:300], z[300:], y[300:], num_classes=4, epochs=100)
    assert acc > 0.95
    noise_acc = train_probe(rng.normal(size=(300, 4)), y[:300], rng.normal(size=(100, 4)), y[300:], num_classes=4, epochs=100)
    assert noise_acc < 0.5


def test_frames_per_episode_caps_frames(tmp_path):
    from worldmodels.vae.dataset import load_frames

    collect_rollouts(tmp_path, num_episodes=10, seed=0)
    paths = episode_paths(tmp_path)
    full = load_frames(paths)
    capped = load_frames(paths, frames_per_episode=2)
    assert len(capped) == 2 * 10 < len(full)


def test_mlp_probe_learns_xor_that_linear_probe_cannot():
    from worldmodels.vae.evaluate import train_probe

    rng = np.random.default_rng(0)
    z = rng.choice([-1.0, 1.0], size=(2000, 2)) + rng.normal(scale=0.1, size=(2000, 2))
    y = ((z[:, 0] > 0) ^ (z[:, 1] > 0)).astype(int)
    args = (z[:1500], y[:1500], z[1500:], y[1500:])
    assert train_probe(*args, num_classes=2, epochs=30) <= 0.8  # beste lineære skille tar 3 av 4 klynger
    assert train_probe(*args, num_classes=2, hidden=16, epochs=30, lr=1e-2) > 0.95
