exec(open("docs/experiments/scripts/handcrafted_goal_seeker.py").read().split("learned =")[0])
waiter = LinearController.load("checkpoints/controller_pos.npz")
for H in [10, 20, 45]:
    for T in [0.0, 1.0]:
        row = []
        for nm, cc in [("mot mål", c), ("venter", waiter)]:
            d = dream_fitness(cc, cc.params[None], rnn, starts, DreamConfig(horizon=H, temperature=T, remaining_steps=45 - H), return_details=True)
            row.append(f"{nm}: {d['fitness'][0]:+.3f} (mål {d['p_goal'][0]:.2f}, hindr {d['p_obstacle'][0]:.2f})")
        print(f"H={H} T={T}  " + "  |  ".join(row), flush=True)
