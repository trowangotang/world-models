import torch

from worldmodels.env.gridworld import ACTIONS
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig
from worldmodels.mdnrnn.tracker import EyeMemory, belief_positions, move_belief, observe, visit_features


def one_hot(cell, cells=64):
    b = torch.zeros(1, cells)
    b[0, cell] = 1.0
    return b


def test_move_belief_follows_the_env_and_stays_at_walls():
    for (r, c) in [(3, 4), (0, 0), (7, 7), (0, 5)]:
        moved = move_belief(one_hot(r * 8 + c))[0]
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = min(max(r + dr, 0), 7), min(max(c + dc, 0), 7)
            assert moved[a].argmax().item() == nr * 8 + nc and torch.isclose(moved[a].sum(), torch.tensor(1.0))


def test_belief_positions_matches_eye_scale():
    assert torch.allclose(belief_positions(one_hot(0)), torch.tensor([[-1.0, -1.0]]))
    assert torch.allclose(belief_positions(one_hot(7 * 8 + 7)), torch.tensor([[1.0, 1.0]]))


class FixedEye(torch.nn.Module):
    """z[:, 0] er cellen øyet ser agenten i, z[:, 1] målet, begge med stor sikkerhet."""
    def forward(self, z):
        logits = torch.full((len(z), 2, 64), -20.0)
        logits[torch.arange(len(z)), 0, z[:, 0].long()] = 20.0
        logits[torch.arange(len(z)), 1, z[:, 1].long()] = 20.0
        return logits


def eye_model():
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4)).eval()
    rnn.eye = FixedEye()
    return rnn


def test_tracker_ignores_a_misreading_that_contradicts_the_action():
    rnn = eye_model()
    z = torch.zeros(1, 4)
    z[0, 1] = 3 * 8 + 1                       # målet står i (3, 1)
    z[0, 0] = 3 * 8 + 3                       # agenten ses i (3, 3)
    _, _, memory = observe(rnn, z, None)
    memory = memory.moved(torch.tensor([2]))  # går til venstre, til (3, 2)
    z[0, 0] = 3 * 8 + 1                       # øyet forveksler agenten med målet
    seen, tracked, memory = observe(rnn, z, memory)
    assert torch.allclose(seen[0, :2], torch.tensor([-1 / 7, -5 / 7]), atol=1e-3)   # øyet alene: (3, 1)
    assert torch.allclose(tracked[0, :2], torch.tensor([-1 / 7, -3 / 7]), atol=0.03)  # filteret: (3, 2)


def test_visit_features_mark_the_way_back_and_the_wall():
    rnn = eye_model()
    z = torch.zeros(1, 4)
    z[0, 1] = 63                              # målet langt unna
    memory = None
    for cell, action in [(8 * 2 + 6, 3), (8 * 2 + 7, None)]:   # (2, 6), høyre til (2, 7) ved veggen
        z[0, 0] = cell
        _, _, memory = observe(rnn, z, memory)
        if action is not None:
            memory = memory.moved(torch.tensor([action]))
    f = visit_features(memory)[0]
    assert f[3] > 0.99            # høyre er veggen: blir stående der den er nå
    assert 0.8 < f[2] < 0.9       # venstre er dit den kom fra, ett skritt glemt
    assert f[0] < 0.05 and f[1] < 0.05


def test_eye_memory_subset_and_repeat():
    m = EyeMemory(torch.randn(3, 64), torch.rand(3, 64), torch.rand(3, 64))
    assert len(m.repeat(2)) == 6 and torch.equal(m.repeat(2).agent[3:], m.agent)
    assert torch.equal(m.subset([2]).visits, m.visits[[2]])


def test_tracker_stays_finite_when_the_agent_seems_to_stand_on_the_goal():
    rnn = eye_model()
    z = torch.zeros(1, 4)
    z[0, 0] = z[0, 1] = 10                    # øyet ser agent og mål i samme celle, igjen og igjen
    memory = None
    for _ in range(5):
        _, tracked, memory = observe(rnn, z, memory)
        memory = memory.moved(torch.tensor([0]))
    assert torch.isfinite(tracked).all() and torch.isfinite(memory.agent).all()
