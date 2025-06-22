import importlib
import math

ppo = importlib.import_module("src.algorithms.ppo")
compute_ppo_loss = ppo.compute_ppo_loss


def test_ppo_loss_clipping_effect():
    old_log_probs = [math.log(0.8), math.log(0.2)]
    new_log_probs = [math.log(0.1), math.log(0.9)]
    adv = [1.0, 1.0]
    returns = [1.0, 1.0]
    values = [0.0, 0.0]

    loss_clip = compute_ppo_loss(
        new_log_probs,
        old_log_probs,
        adv,
        returns,
        values,
        clip_range=0.2,
    )

    ratio = [math.exp(n - o) for n, o in zip(new_log_probs, old_log_probs)]
    policy_terms = [r * a for r, a in zip(ratio, adv)]
    policy_loss_no_clip = -sum(policy_terms) / len(policy_terms)
    value_terms = [(r - v) ** 2 for r, v in zip(returns, values)]
    value_loss = 0.5 * sum(value_terms) / len(value_terms)
    entropy_terms = [-math.exp(n) * n for n in new_log_probs]
    entropy = sum(entropy_terms) / len(entropy_terms)
    loss_no_clip = policy_loss_no_clip + 0.5 * value_loss - 0.01 * entropy

    assert abs(loss_clip - loss_no_clip) > 1e-6
