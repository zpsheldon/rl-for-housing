"""
train_reinforce.py

REINFORCE with learned baseline (policy gradient) for the HousingEnv.

The environment's hierarchical action space is Discrete(num_inspectors *
max_active_reports).  Because all inspectors are always available and produce
identical outcome distributions, the meaningful decision is *which report* to
inspect.  The policy therefore outputs a distribution over max_active_reports
report slots, and inspector assignment is done at random when the environment
action is constructed.  This reduces the output layer from ~90 000 to 600
neurons while preserving correctness.

Callable from a notebook:

    from train_reinforce import train_and_evaluate
    results = train_and_evaluate(num_episodes=50, verbose=True)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from housing_env import HousingEnv, RewardWeights


# ── Observation utilities ──────────────────────────────────────────────────────

def flatten_obs(obs: dict) -> np.ndarray:
    """Convert the Dict observation to a 1-D float32 array.

    Matches the layout used by FlattenDictObsWrapper in run_models.py:
        reports (max_active_reports × 10) | mask (max_active_reports) |
        inspectors (num_inspectors) | timestep (1)
    """
    flat = np.concatenate([
        obs["reports"].flatten().astype(np.float32),
        obs["mask"].astype(np.float32),
        np.atleast_1d(np.asarray(obs["inspectors"], dtype=np.float32)).flatten(),
        np.atleast_1d(np.asarray(obs["timestep"],   dtype=np.float32)).flatten(),
    ])
    # Guard against NaN/inf that can arise from missing values in the dataset
    # (e.g. Council District being NaN) — replace with 0 so the network sees
    # a valid zero-padded feature rather than propagating NaN through gradients.
    return np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)


def obs_dim_for(env: HousingEnv) -> int:
    """Compute the flat observation dimension for a given env."""
    r = env.observation_space["reports"].shape      # (max_active_reports, 10)
    return r[0] * r[1] + r[0] + env.num_inspectors + 1


def report_mask_from(obs: dict) -> np.ndarray:
    """Return a boolean mask of shape (max_active_reports,) — True = slot active."""
    return obs["mask"].astype(bool)


# ── Networks ───────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Stochastic policy that selects which report to inspect.

    Outputs a categorical distribution over the max_active_reports report
    slots.  Invalid (empty) slots are masked to -inf before sampling.
    """

    def __init__(self, obs_dim: int, max_active_reports: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_active_reports),
        )

    def forward(self, x: torch.Tensor, report_mask: torch.Tensor | None = None) -> Categorical:
        """
        Args:
            x:           FloatTensor (batch, obs_dim)
            report_mask: BoolTensor  (batch, max_active_reports) — True = valid
        Returns:
            Categorical distribution over report slots
        """
        logits = self.net(x)
        if report_mask is not None:
            logits = logits.masked_fill(~report_mask, -1e9)
        # Clamp to prevent overflow to ±inf that would cause Categorical to
        # raise ValueError ("logits must satisfy the Real() constraint").
        logits = logits.clamp(min=-1e9, max=1e9)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """State-value baseline V(s).  Trained to predict discounted returns."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── REINFORCE with baseline agent ──────────────────────────────────────────────

class REINFORCEAgent:
    """REINFORCE with a learned value-function baseline.

    After each complete episode, the agent computes discounted returns G_t,
    subtracts the baseline V(s_t) to form advantages, and updates:

        policy loss = -E[ (G_t - V(s_t)) * log π(a_t | s_t) ]
        value  loss =  E[ (G_t - V(s_t))^2 ]

    Steps where no reports are active (mask all-zero) still contribute to the
    return calculation but are excluded from the policy / value gradient via
    separate tracking.
    """

    def __init__(
        self,
        obs_dim: int,
        max_active_reports: int,
        num_inspectors: int,
        hidden_dim: int = 256,
        lr_policy: float = 1e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        normalize_returns: bool = True,
        device: str = "cpu",
    ):
        self.num_inspectors = num_inspectors
        self.max_active_reports = max_active_reports
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.device = device

        self.policy = PolicyNetwork(obs_dim, max_active_reports, hidden_dim).to(device)
        self.value  = ValueNetwork(obs_dim, hidden_dim).to(device)

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_opt  = optim.Adam(self.value.parameters(),  lr=lr_value)

        # Episode trajectory buffers
        self._log_probs  : list[torch.Tensor] = []
        self._values     : list[torch.Tensor] = []
        self._rewards    : list[float]        = []
        # Maps each policy step i → reward-buffer index so returns are computed
        # over the full reward sequence (including no-op steps).
        self._step_map   : list[int]          = []

    # ------------------------------------------------------------------

    def _reset_networks(self) -> None:
        """Reinitialise both networks and clear episode buffers.

        Called when NaN is detected in parameters so that training can
        continue from a clean state rather than propagating corruption.
        """
        for net in (self.policy, self.value):
            net.apply(
                lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
            )
        self._log_probs, self._values, self._rewards, self._step_map = [], [], [], []

    def select_action(self, obs_flat: np.ndarray, report_mask_np: np.ndarray) -> int:
        """Sample a report, record log-prob and value, return env action.

        The env action encodes (inspector_id, report_id) as
            inspector_id * max_active_reports + report_id
        Inspector assignment is random because all inspectors are identical
        in this environment (always available, same outcome distributions).

        Args:
            obs_flat       : 1-D float32 observation vector
            report_mask_np : bool array (max_active_reports,), True = valid slot

        Returns:
            int: action for env.step()
        """
        # If weights have become NaN (e.g. from a previous corrupted update),
        # reinitialise both networks so training can recover cleanly.
        if any(not torch.isfinite(p).all() for p in self.policy.parameters()):
            self._reset_networks()

        obs_t  = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(report_mask_np).unsqueeze(0).to(self.device)

        dist      = self.policy(obs_t, mask_t)
        value     = self.value(obs_t)
        report_id = dist.sample()

        # Record mapping: this policy step corresponds to the next reward slot
        self._step_map.append(len(self._rewards))
        self._log_probs.append(dist.log_prob(report_id))
        self._values.append(value)

        inspector_id = np.random.randint(0, self.num_inspectors)
        return int(inspector_id * self.max_active_reports + report_id.item())

    def store_reward(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def update(self) -> dict:
        """Perform REINFORCE-with-baseline update after a complete episode.

        Returns a dict with 'policy_loss', 'value_loss', and 'total_reward'.
        """
        T = len(self._rewards)
        n_valid = len(self._log_probs)

        if n_valid == 0:
            self._rewards = []
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_reward": 0.0}

        # ── Compute discounted returns for all reward steps ────────────────
        all_returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            all_returns[t] = G

        # ── Extract returns at the valid policy-step positions ─────────────
        returns = torch.stack([all_returns[i] for i in self._step_map])

        if self.normalize_returns and n_valid > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── Stack stored tensors ───────────────────────────────────────────
        log_probs = torch.stack(self._log_probs)              # (n_valid,)
        values    = torch.stack(self._values).squeeze(-1)     # (n_valid,)

        # Sanitize: replace any residual NaN/inf so a single bad observation
        # can never permanently corrupt the network weights.
        returns   = torch.nan_to_num(returns,   nan=0.0, posinf=1e6, neginf=-1e6)
        values    = torch.nan_to_num(values,    nan=0.0, posinf=1e6, neginf=-1e6)
        log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=-1e6)

        advantages = returns - values.detach()

        # ── Losses ────────────────────────────────────────────────────────
        policy_loss = -(log_probs * advantages).mean()
        value_loss  = nn.functional.mse_loss(values, returns.detach())

        # Guard: skip weight update if losses are still non-finite (e.g. due
        # to exploding returns on a degenerate episode).
        if not (torch.isfinite(policy_loss) and torch.isfinite(value_loss)):
            total_reward = sum(self._rewards)
            self._log_probs, self._values, self._rewards, self._step_map = [], [], [], []
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_reward": total_reward}

        # ── Gradient steps ────────────────────────────────────────────────
        self.policy_opt.zero_grad()
        policy_loss.backward()
        for p in self.policy.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        for p in self.value.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        self.value_opt.step()

        total_reward = sum(self._rewards)

        # ── Clear buffers ─────────────────────────────────────────────────
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._step_map  = []

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "total_reward": total_reward,
        }

    def predict(self, obs_flat: np.ndarray, report_mask_np: np.ndarray) -> int:
        """Greedy (deterministic) action selection for evaluation."""
        obs_t  = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(report_mask_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist      = self.policy(obs_t, mask_t)
            report_id = dist.probs.argmax().item()
        inspector_id = np.random.randint(0, self.num_inspectors)
        return int(inspector_id * self.max_active_reports + report_id)

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict(), "value": self.value.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _step_metrics(info: dict) -> tuple[int, int]:
    """Extract (violations_this_step, reports_closed_this_step) from step info."""
    details = info.get("inspection_details", [])
    viol = sum(1 for d in details if d.get("outcome") == "VIOLATION")
    return viol, len(details)


def evaluate_agent(
    agent: REINFORCEAgent,
    env: HousingEnv,
    max_steps: int,
    verbose: bool = True,
) -> dict:
    """Roll out the REINFORCE agent deterministically on env.

    Returns a dict with per-step lists: rewards, violations, reports_closed,
    and a scalar open_reports at episode end.
    """
    obs, _ = env.reset()
    rewards, violations, reports_closed = [], [], []
    open_reports = 0

    for t in range(max_steps):
        rmask = report_mask_from(obs)
        if rmask.any():
            action = agent.predict(flatten_obs(obs), rmask)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        v, c = _step_metrics(info)
        violations.append(v)
        reports_closed.append(c)
        open_reports = info.get("open_reports", 0)

        if verbose and (t + 1) % 200 == 0:
            print(f"  REINFORCE eval step {t+1}/{max_steps}")

        if terminated or truncated:
            break

    return {"rewards": rewards, "violations": violations,
            "reports_closed": reports_closed, "open_reports": open_reports}


def evaluate_random(env: HousingEnv, max_steps: int, verbose: bool = True) -> dict:
    """Roll out a random agent for comparison."""
    obs, _ = env.reset()
    rewards, violations, reports_closed = [], [], []
    open_reports = 0

    for t in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        rewards.append(reward)
        v, c = _step_metrics(info)
        violations.append(v)
        reports_closed.append(c)
        open_reports = info.get("open_reports", 0)

        if verbose and (t + 1) % 200 == 0:
            print(f"  Random eval step {t+1}/{max_steps}")

        if terminated or truncated:
            break

    return {"rewards": rewards, "violations": violations,
            "reports_closed": reports_closed, "open_reports": open_reports}


# ── Visualisation ──────────────────────────────────────────────────────────────

def _plot_results(
    training_history: list[dict],
    reinforce_eval: dict,
    random_eval: dict,
    save_path: str = "reinforce-comparison.png",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("REINFORCE with Baseline — Housing Inspection RL", fontsize=14)

    # Panel 1: training curve
    ax = axes[0, 0]
    ep_rewards = [h["total_reward"] for h in training_history]
    window = max(1, len(ep_rewards) // 10)
    rolling = pd.Series(ep_rewards).rolling(window, min_periods=1).mean()
    ax.plot(ep_rewards, alpha=0.35, color="steelblue")
    ax.plot(rolling, color="steelblue", linewidth=2, label=f"Rolling mean ({window} ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: cumulative reward (eval)
    ax = axes[0, 1]
    ax.plot(np.cumsum(random_eval["rewards"]),    label="Random Baseline",         linewidth=2)
    ax.plot(np.cumsum(reinforce_eval["rewards"]), label="REINFORCE with Baseline",  linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward (Evaluation Year)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: cumulative violations (eval)
    ax = axes[1, 0]
    ax.plot(np.cumsum(random_eval["violations"]),    label="Random Baseline",        linewidth=2)
    ax.plot(np.cumsum(reinforce_eval["violations"]), label="REINFORCE with Baseline", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Violations Fixed")
    ax.set_title("Violations Fixed (Evaluation Year)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: normalised bar chart
    ax = axes[1, 1]
    labels   = ["Total Reward", "Violations Fixed", "Reports Closed"]
    b_vals   = [sum(random_eval["rewards"]),    sum(random_eval["violations"]),    sum(random_eval["reports_closed"])]
    r_vals   = [sum(reinforce_eval["rewards"]), sum(reinforce_eval["violations"]), sum(reinforce_eval["reports_closed"])]
    norms    = [max(b_vals[i], r_vals[i]) or 1 for i in range(3)]
    b_norm   = [b_vals[i] / norms[i] for i in range(3)]
    r_norm   = [r_vals[i] / norms[i] for i in range(3)]
    x, w = np.arange(3), 0.35
    ax.bar(x - w / 2, b_norm, w, label="Random Baseline",         alpha=0.8)
    ax.bar(x + w / 2, r_norm, w, label="REINFORCE with Baseline", alpha=0.8)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Performance Comparison (Normalized, Eval Year)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")


# ── Results persistence ────────────────────────────────────────────────────────

def save_results(
    output_dir: str = "results",
    training_history: list[dict] | None = None,
    reinforce_eval: dict | None = None,
    random_eval: dict | None = None,
    dqn_rewards: list | None = None,
    dqn_violations: list | None = None,
    dqn_reports_closed: list | None = None,
    comparison: "pd.DataFrame | None" = None,
) -> None:
    """Save training and evaluation results to CSVs under output_dir.

    Files written:
        training_history.csv  — per-episode REINFORCE training metrics
        eval_reinforce.csv    — per-step REINFORCE evaluation
        eval_random.csv       — per-step random baseline evaluation
        eval_dqn.csv          — per-step DQN evaluation
        summary_comparison.csv — aggregate comparison table

    Only files whose source data is provided are written; existing files in
    output_dir are overwritten silently.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _eval_to_df(eval_dict: dict) -> pd.DataFrame:
        rewards   = eval_dict["rewards"]
        viol      = eval_dict["violations"]
        closed    = eval_dict["reports_closed"]
        return pd.DataFrame({
            "step":                   range(len(rewards)),
            "reward":                 rewards,
            "cumulative_reward":      list(np.cumsum(rewards)),
            "violations":             viol,
            "cumulative_violations":  list(np.cumsum(viol)),
            "reports_closed":         closed,
            "cumulative_reports_closed": list(np.cumsum(closed)),
        })

    if training_history is not None:
        pd.DataFrame(training_history).to_csv(
            os.path.join(output_dir, "training_history.csv"), index=False
        )
        print(f"Saved training_history.csv  ({len(training_history)} episodes)")

    if reinforce_eval is not None:
        _eval_to_df(reinforce_eval).to_csv(
            os.path.join(output_dir, "eval_reinforce.csv"), index=False
        )
        print(f"Saved eval_reinforce.csv    ({len(reinforce_eval['rewards'])} steps)")

    if random_eval is not None:
        _eval_to_df(random_eval).to_csv(
            os.path.join(output_dir, "eval_random.csv"), index=False
        )
        print(f"Saved eval_random.csv       ({len(random_eval['rewards'])} steps)")

    if dqn_rewards is not None:
        n = len(dqn_rewards)
        dqn_viol   = dqn_violations   or [0] * n
        dqn_closed = dqn_reports_closed or [0] * n
        pd.DataFrame({
            "step":                      range(n),
            "reward":                    dqn_rewards,
            "cumulative_reward":         list(np.cumsum(dqn_rewards)),
            "violations":                dqn_viol,
            "cumulative_violations":     list(np.cumsum(dqn_viol)),
            "reports_closed":            dqn_closed,
            "cumulative_reports_closed": list(np.cumsum(dqn_closed)),
        }).to_csv(os.path.join(output_dir, "eval_dqn.csv"), index=False)
        print(f"Saved eval_dqn.csv          ({n} steps)")

    if comparison is not None:
        comparison.to_csv(
            os.path.join(output_dir, "summary_comparison.csv"), index=False
        )
        print(f"Saved summary_comparison.csv")

    print(f"\nAll results saved to '{output_dir}/'")


def load_results(output_dir: str = "results") -> dict:
    """Load previously saved results from output_dir.

    Returns a dict with keys:
        training_history      pd.DataFrame or None
        eval_reinforce        pd.DataFrame or None
        eval_random           pd.DataFrame or None
        eval_dqn              pd.DataFrame or None
        summary_comparison    pd.DataFrame or None

    Missing files produce None values rather than raising errors.
    """
    files = {
        "training_history":   "training_history.csv",
        "eval_reinforce":     "eval_reinforce.csv",
        "eval_random":        "eval_random.csv",
        "eval_dqn":           "eval_dqn.csv",
        "summary_comparison": "summary_comparison.csv",
    }
    loaded = {}
    for key, fname in files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            loaded[key] = pd.read_csv(path)
            print(f"Loaded {fname}  ({len(loaded[key])} rows)")
        else:
            loaded[key] = None
    return loaded


# ── Main entry point ───────────────────────────────────────────────────────────

def train_and_evaluate(
    data_path: str = "data/311_preproc.csv",
    num_inspectors: int = 150,
    max_active_reports: int = 600,
    inspection_rate: int = 4,
    train_years: int = 4,
    test_years: int = 1,
    train_start_date: str | None = None,
    test_start_date: str | None = None,
    num_episodes: int = 50,
    gamma: float = 0.99,
    lr_policy: float = 1e-4,
    lr_value: float = 1e-3,
    hidden_dim: int = 256,
    normalize_returns: bool = True,
    device: str = "auto",
    seed: int = 42,
    save_path: str | None = "models/reinforce_baseline.pt",
    plot: bool = True,
    verbose: bool = True,
    reward_weights: RewardWeights | None = None,
) -> dict:
    """Train REINFORCE with baseline on HousingEnv, then evaluate vs. random.

    Parameters
    ----------
    data_path            Path to the preprocessed 311 CSV.
    num_inspectors       Number of inspectors in the environment.
    max_active_reports   Maximum active complaint pool size.
    inspection_rate      Inspections per inspector per day (env param).
    train_years          Duration of each training episode in years.
    test_years           Duration of the evaluation episode in years.
    train_start_date     ISO date for training start; auto-detected if None.
    test_start_date      ISO date for test start; auto-detected if None.
    num_episodes         Number of training episodes.
    gamma                Discount factor.
    lr_policy            Learning rate for the policy network.
    lr_value             Learning rate for the value (baseline) network.
    hidden_dim           Hidden layer width for both networks.
    normalize_returns    Normalise episode returns before each update.
    device               'auto', 'cpu', or 'cuda'.
    seed                 Random seed.
    save_path            Checkpoint path (None to skip saving).
    plot                 Generate and display comparison plots.
    verbose              Print training progress.

    Returns
    -------
    dict with keys:
        "agent"            : trained REINFORCEAgent
        "training_history" : list of per-episode metric dicts
        "reinforce_eval"   : per-step eval results (rewards, violations, …)
        "random_eval"      : per-step eval results for random baseline
        "comparison"       : summary comparison DataFrame
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Auto-detect dates from data ────────────────────────────────────────
    df_dates = pd.read_csv(data_path, usecols=["Created Date"])
    df_dates["Created Date"] = pd.to_datetime(df_dates["Created Date"])
    min_date = df_dates["Created Date"].min()

    if train_start_date is None:
        train_start_date = str(min_date.date())
    if test_start_date is None:
        test_start_date = str(
            (pd.to_datetime(train_start_date) + pd.DateOffset(years=train_years)).date()
        )

    if verbose:
        print(f"{'='*60}")
        print("REINFORCE WITH BASELINE — HousingEnv")
        print(f"{'='*60}")
        print(f"  Training : {train_start_date}  ({train_years} yr × {num_episodes} episodes)")
        print(f"  Testing  : {test_start_date}  ({test_years} yr)")
        print(f"  Device   : {device}")

    # ── Determine obs/action dimensions ───────────────────────────────────
    probe = HousingEnv(
        num_inspectors=num_inspectors,
        inspection_rate=inspection_rate,
        years=train_years,
        max_active_reports=max_active_reports,
        hierarchical=True,
        data_path=data_path,
        start_date_str=train_start_date,
        reward_weights=reward_weights,
    )
    obs_dim = obs_dim_for(probe)
    probe.close()

    if verbose:
        print(f"  Obs dim  : {obs_dim}  |  Policy outputs: {max_active_reports}"
              f"  |  Env action space: Discrete({num_inspectors * max_active_reports})")
        print()

    # ── Build agent ────────────────────────────────────────────────────────
    agent = REINFORCEAgent(
        obs_dim=obs_dim,
        max_active_reports=max_active_reports,
        num_inspectors=num_inspectors,
        hidden_dim=hidden_dim,
        lr_policy=lr_policy,
        lr_value=lr_value,
        gamma=gamma,
        normalize_returns=normalize_returns,
        device=device,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    training_history: list[dict] = []
    t0 = time.time()
    log_every = max(1, num_episodes // 10)

    for ep in range(1, num_episodes + 1):
        env = HousingEnv(
            num_inspectors=num_inspectors,
            inspection_rate=inspection_rate,
            years=train_years,
            max_active_reports=max_active_reports,
            hierarchical=True,
            data_path=data_path,
            start_date_str=train_start_date,
            reward_weights=reward_weights,
        )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_violations = 0
        ep_closed     = 0

        while not done:
            rmask = report_mask_from(obs)
            if rmask.any():
                action = agent.select_action(flatten_obs(obs), rmask)
            else:
                action = 0  # no-op; reward still stored below

            obs, reward, terminated, truncated, info = env.step(action)
            agent.store_reward(reward)

            v, c = _step_metrics(info)
            ep_violations += v
            ep_closed     += c
            done = terminated or truncated

        env.close()

        metrics = agent.update()
        metrics.update(episode=ep, violations_fixed=ep_violations, reports_closed=ep_closed)
        training_history.append(metrics)

        if verbose and ep % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  Episode {ep:4d}/{num_episodes}"
                f"  |  Reward: {metrics['total_reward']:8.1f}"
                f"  |  Violations: {ep_violations:4d}"
                f"  |  PolicyLoss: {metrics['policy_loss']:.4f}"
                f"  |  ValueLoss: {metrics['value_loss']:.4f}"
                f"  |  {elapsed:.0f}s elapsed"
            )

    # ── Save checkpoint ────────────────────────────────────────────────────
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        agent.save(save_path)
        if verbose:
            print(f"\nModel saved to {save_path}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    # Use the same loop-bound convention as run_models.py:
    #   int(test_years * 365 * inspection_rate) gives an upper bound;
    #   the env terminates early at test_years * 365 steps.
    eval_max_steps = int(test_years * 365 * inspection_rate)

    if verbose:
        print(f"\nEvaluating REINFORCE agent on test period ({test_start_date})...")
    env_reinforce = HousingEnv(
        num_inspectors=num_inspectors, inspection_rate=inspection_rate,
        years=test_years, max_active_reports=max_active_reports,
        hierarchical=True, data_path=data_path, start_date_str=test_start_date,
        reward_weights=reward_weights,
    )
    reinforce_eval = evaluate_agent(agent, env_reinforce, eval_max_steps, verbose=verbose)
    env_reinforce.close()

    if verbose:
        print(f"\nEvaluating random baseline on test period ({test_start_date})...")
    env_random = HousingEnv(
        num_inspectors=num_inspectors, inspection_rate=inspection_rate,
        years=test_years, max_active_reports=max_active_reports,
        hierarchical=True, data_path=data_path, start_date_str=test_start_date,
        reward_weights=reward_weights,
    )
    random_eval = evaluate_random(env_random, eval_max_steps, verbose=verbose)
    env_random.close()

    # ── Summary table ──────────────────────────────────────────────────────
    r_total  = sum(reinforce_eval["rewards"])
    r_viol   = sum(reinforce_eval["violations"])
    r_closed = sum(reinforce_eval["reports_closed"])
    r_open   = reinforce_eval["open_reports"]
    r_avg    = float(np.mean(reinforce_eval["rewards"]))

    b_total  = sum(random_eval["rewards"])
    b_viol   = sum(random_eval["violations"])
    b_closed = sum(random_eval["reports_closed"])
    b_open   = random_eval["open_reports"]
    b_avg    = float(np.mean(random_eval["rewards"]))

    comparison = pd.DataFrame({
        "Metric": ["Total Reward", "Violations Fixed", "Reports Closed",
                   "Final Open Reports", "Avg Reward/Step"],
        "Random Baseline": [f"{b_total:.2f}", f"{b_viol}", f"{b_closed}",
                            f"{b_open}", f"{b_avg:.4f}"],
        "REINFORCE":       [f"{r_total:.2f}", f"{r_viol}", f"{r_closed}",
                            f"{r_open}", f"{r_avg:.4f}"],
    })

    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON: RANDOM BASELINE vs. REINFORCE WITH BASELINE")
        print("=" * 60)
        print(comparison.to_string(index=False))
        rwd_pct  = (r_total - b_total) / abs(b_total) * 100 if b_total else 0
        viol_pct = (r_viol  - b_viol)  / max(b_viol, 1)  * 100
        print(f"\nREINFORCE vs. Random:  Reward {rwd_pct:+.1f}%  |  Violations {viol_pct:+.1f}%")

    # ── Plots ──────────────────────────────────────────────────────────────
    if plot:
        _plot_results(training_history, reinforce_eval, random_eval)

    return {
        "agent":            agent,
        "training_history": training_history,
        "reinforce_eval":   reinforce_eval,
        "random_eval":      random_eval,
        "comparison":       comparison,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate(
        num_episodes=50,
        train_years=4,
        test_years=1,
        num_inspectors=150,
        max_active_reports=600,
        verbose=True,
    )
