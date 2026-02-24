"""Hybrid GA-RL Agent for Inventory Optimization.

Two-phase strategy:
    Phase 1 — Genetic Algorithm discovers a strong static reorder policy by
              evolving threshold vectors over multiple generations.
    Phase 2 — Q-Learning fine-tunes the policy online, starting from a warm
              Q-table seeded by the GA solution (lower exploration needed).

Designed for :class:`app.rl.environment.InventoryEnv` which exposes:
    observation_space : Box(5,) float32 [0, 1]
    action_space      : Discrete(5) — 0=no order, 1-4 = EOQ multiples
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genetic Algorithm optimizer
# ---------------------------------------------------------------------------


class GAOptimizer:
    """Evolve a population of deterministic reorder policies.

    Each *individual* (gene) is a 1-D numpy array of shape ``(n_genes,)``
    representing normalized thresholds::

        gene = [stock_threshold, pending_threshold, demand_threshold]

    The deterministic policy derived from a gene works as follows:

    1. If observed stock < stock_threshold **and** pending orders <
       pending_threshold **and** demand trend > demand_threshold, then
       the agent places an order whose size is proportional to the
       perceived deficit (mapped to actions 1-4).
    2. Otherwise the agent does nothing (action 0).

    Parameters
    ----------
    pop_size : int
        Number of individuals per generation.
    n_genes : int
        Dimensionality of the threshold vector.
    mutation_rate : float
        Probability that any single gene is mutated during reproduction.
    crossover_rate : float
        Probability that a pair of parents undergo single-point crossover.
    generations : int
        Number of evolutionary generations to run.
    tournament_k : int
        Tournament selection pool size.
    seed : int or None
        Reproducibility seed for the internal RNG.
    """

    def __init__(
        self,
        pop_size: int = 50,
        n_genes: int = 3,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        generations: int = 40,
        tournament_k: int = 3,
        seed: int | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.tournament_k = tournament_k
        self.rng = np.random.default_rng(seed)

        # Diagnostics populated by ``run``
        self.best_gene: np.ndarray | None = None
        self.best_fitness: float = -np.inf
        self.history: list[dict[str, Any]] = []

    # -- deterministic policy from gene -----------------------------------

    @staticmethod
    def _policy_from_gene(gene: np.ndarray, obs: np.ndarray) -> int:
        """Map a threshold gene + observation to a discrete action.

        Parameters
        ----------
        gene : ndarray, shape (3,)
            [stock_threshold, pending_threshold, demand_threshold]
        obs : ndarray, shape (5,)
            [stock_norm, pending_norm, days_since_norm, demand_trend, stockout_norm]

        Returns
        -------
        int
            Action in {0, 1, 2, 3, 4}.
        """
        stock_norm = obs[0]
        pending_norm = obs[1]
        demand_trend = obs[3]

        stock_thresh, pending_thresh, demand_thresh = gene

        # Decision: should we reorder?
        needs_reorder = (
            stock_norm < stock_thresh
            and pending_norm < pending_thresh
            and demand_trend > demand_thresh
        )

        if not needs_reorder:
            return 0

        # Determine order magnitude from the deficit size.
        # Larger deficit (lower stock relative to threshold) -> bigger order.
        deficit_ratio = np.clip((stock_thresh - stock_norm) / max(stock_thresh, 1e-6), 0.0, 1.0)

        if deficit_ratio > 0.75:
            return 4  # 2.0 x EOQ
        elif deficit_ratio > 0.50:
            return 3  # 1.5 x EOQ
        elif deficit_ratio > 0.25:
            return 2  # 1.0 x EOQ
        else:
            return 1  # 0.5 x EOQ

    # -- fitness evaluation -----------------------------------------------

    def evaluate(self, individual: np.ndarray, env: Any) -> float:
        """Run one full episode using the deterministic gene policy.

        Parameters
        ----------
        individual : ndarray, shape (n_genes,)
        env : InventoryEnv (Gymnasium API)

        Returns
        -------
        float
            Cumulative (undiscounted) reward for the episode.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self._policy_from_gene(individual, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        return total_reward

    # -- selection / crossover / mutation ---------------------------------

    def _tournament_select(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        """Tournament selection — pick the best among *k* random candidates."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def _crossover(
        self, parent_a: np.ndarray, parent_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-point crossover with probability ``crossover_rate``."""
        if self.rng.random() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        point = self.rng.integers(1, self.n_genes)  # at least 1 gene swapped
        child_a = np.concatenate([parent_a[:point], parent_b[point:]])
        child_b = np.concatenate([parent_b[:point], parent_a[point:]])
        return child_a, child_b

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation — each gene independently perturbed."""
        mutant = individual.copy()
        for i in range(self.n_genes):
            if self.rng.random() < self.mutation_rate:
                mutant[i] += self.rng.normal(0.0, 0.1)
                mutant[i] = np.clip(mutant[i], 0.0, 1.0)
        return mutant

    # -- main loop --------------------------------------------------------

    def run(self, env: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Execute the full GA optimisation loop.

        Parameters
        ----------
        env : InventoryEnv

        Returns
        -------
        best_gene : ndarray, shape (n_genes,)
            Best threshold vector found across all generations.
        history : list of dict
            Per-generation statistics (best / mean / worst fitness).
        """
        logger.info(
            "GA starting: pop=%d, gens=%d, mut=%.2f, cx=%.2f",
            self.pop_size,
            self.generations,
            self.mutation_rate,
            self.crossover_rate,
        )
        t0 = time.perf_counter()

        # Initialize population uniformly in [0, 1]
        population = self.rng.random((self.pop_size, self.n_genes))

        self.best_fitness = -np.inf
        self.best_gene = population[0].copy()
        self.history = []

        for gen in range(self.generations):
            # Evaluate
            fitnesses = np.array(
                [self.evaluate(ind, env) for ind in population]
            )

            # Track best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            gen_mean_fit = float(np.mean(fitnesses))
            gen_worst_fit = float(np.min(fitnesses))

            if gen_best_fit > self.best_fitness:
                self.best_fitness = gen_best_fit
                self.best_gene = population[gen_best_idx].copy()

            gen_stats = {
                "generation": gen,
                "best_fitness": float(gen_best_fit),
                "mean_fitness": gen_mean_fit,
                "worst_fitness": gen_worst_fit,
                "global_best_fitness": float(self.best_fitness),
            }
            self.history.append(gen_stats)

            if gen % 10 == 0 or gen == self.generations - 1:
                logger.info(
                    "GA gen %3d/%d  best=%.1f  mean=%.1f  global_best=%.1f",
                    gen,
                    self.generations,
                    gen_best_fit,
                    gen_mean_fit,
                    self.best_fitness,
                )

            # Elitism: carry over the single best individual
            next_population = [population[gen_best_idx].copy()]

            # Breed the rest
            while len(next_population) < self.pop_size:
                parent_a = self._tournament_select(population, fitnesses)
                parent_b = self._tournament_select(population, fitnesses)
                child_a, child_b = self._crossover(parent_a, parent_b)
                child_a = self._mutate(child_a)
                child_b = self._mutate(child_b)
                next_population.append(child_a)
                if len(next_population) < self.pop_size:
                    next_population.append(child_b)

            population = np.array(next_population)

        elapsed = time.perf_counter() - t0
        logger.info(
            "GA finished in %.1fs — best fitness=%.1f  gene=%s",
            elapsed,
            self.best_fitness,
            np.array2string(self.best_gene, precision=3),
        )

        return self.best_gene, self.history


# ---------------------------------------------------------------------------
# Hybrid GA + Q-Learning agent
# ---------------------------------------------------------------------------


class HybridGARLAgent:
    """Two-phase inventory agent: GA warm-start followed by Q-Learning.

    Phase 1 (offline):
        A :class:`GAOptimizer` searches for a strong static reorder policy.
        The resulting threshold gene is used to *bias* the initial Q-table so
        that the preferred GA action receives a higher starting value.

    Phase 2 (online):
        Standard tabular Q-Learning with epsilon-greedy exploration.  Because
        the Q-table is warm-started, epsilon begins much lower (default 0.3)
        than a cold-start agent would need.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space (default 5 for InventoryEnv).
    n_bins : int
        Number of uniform bins per observation dimension when discretising
        the continuous state into a flat index for the Q-table.
    ga_pop : int
        GA population size.
    ga_gens : int
        GA generation count.
    rl_alpha : float
        Q-Learning step-size (learning rate).
    rl_gamma : float
        Discount factor.
    rl_epsilon_start : float
        Initial exploration rate for epsilon-greedy (post GA warm-start).
    rl_epsilon_min : float
        Minimum exploration rate.
    rl_epsilon_decay : float
        Multiplicative decay applied to epsilon after each episode.
    seed : int or None
        Reproducibility seed.
    """

    def __init__(
        self,
        n_actions: int = 5,
        n_bins: int = 10,
        ga_pop: int = 50,
        ga_gens: int = 40,
        rl_alpha: float = 0.1,
        rl_gamma: float = 0.99,
        rl_epsilon_start: float = 0.3,
        rl_epsilon_min: float = 0.01,
        rl_epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.n_obs_dims = 5  # InventoryEnv observation dimension

        # GA hyper-parameters
        self.ga_pop = ga_pop
        self.ga_gens = ga_gens

        # RL hyper-parameters
        self.alpha = rl_alpha
        self.gamma = rl_gamma
        self.epsilon = rl_epsilon_start
        self.epsilon_start = rl_epsilon_start
        self.epsilon_min = rl_epsilon_min
        self.epsilon_decay = rl_epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Q-table: rows = discretised states, columns = actions
        n_states = n_bins ** self.n_obs_dims
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        # Tracking
        self._ga_gene: np.ndarray | None = None
        self._ga_history: list[dict[str, Any]] = []
        self._ga_fitness: float = -np.inf
        self._ga_elapsed: float = 0.0
        self._rl_episode_count: int = 0
        self._rl_step_count: int = 0
        self._rl_episode_rewards: list[float] = []
        self._current_episode_reward: float = 0.0

        logger.info(
            "HybridGARLAgent created: n_actions=%d, n_bins=%d, "
            "Q-table shape=%s, epsilon_start=%.2f",
            n_actions,
            n_bins,
            self.q_table.shape,
            rl_epsilon_start,
        )

    # -- state discretisation ---------------------------------------------

    def _discretise(self, obs: np.ndarray) -> int:
        """Convert a continuous [0,1]^5 observation to a flat Q-table index.

        Each dimension is uniformly binned into ``n_bins`` intervals.
        The flat index is computed via mixed-radix encoding.
        """
        clipped = np.clip(obs, 0.0, 1.0)
        bins = np.clip(
            (clipped * self.n_bins).astype(int), 0, self.n_bins - 1
        )
        flat = 0
        for b in bins:
            flat = flat * self.n_bins + int(b)
        return flat

    # -- Phase 1: GA warm-start -------------------------------------------

    def initialize_from_ga(self, env: Any) -> dict[str, Any]:
        """Run the GA optimiser and seed the Q-table from the best gene.

        The best gene's deterministic policy is evaluated across a sweep of
        representative states.  For each discretised state, the preferred
        action receives a positive bias in the Q-table while the remaining
        actions stay at zero.  This gives Q-Learning a head start.

        Parameters
        ----------
        env : InventoryEnv

        Returns
        -------
        dict
            Summary of the GA phase (best gene, fitness, time).
        """
        logger.info("Phase 1: running GA optimisation for warm-start ...")
        t0 = time.perf_counter()

        ga = GAOptimizer(
            pop_size=self.ga_pop,
            n_genes=3,
            mutation_rate=0.15,
            crossover_rate=0.8,
            generations=self.ga_gens,
            seed=int(self.rng.integers(0, 2**31)) if self.rng is not None else None,
        )
        best_gene, history = ga.run(env)

        self._ga_gene = best_gene
        self._ga_history = history
        self._ga_fitness = ga.best_fitness
        self._ga_elapsed = time.perf_counter() - t0

        # Seed Q-table --------------------------------------------------
        # Generate representative observations by sweeping a coarse grid
        # over the 5-D normalised space and writing the GA-preferred action
        # into the corresponding Q-table cell.
        bias_value = abs(self._ga_fitness) * 0.1 if self._ga_fitness != 0 else 1.0
        bias_value = max(bias_value, 1.0)  # ensure a meaningful bias

        seeded_states = 0
        grid_points = np.linspace(0.05, 0.95, min(self.n_bins, 6))

        for s0 in grid_points:          # stock
            for s1 in grid_points:      # pending
                for s3 in grid_points:  # demand_trend
                    # Fix days_since_order and stockout_days at moderate values
                    for s2 in [0.2, 0.5, 0.8]:
                        for s4 in [0.0, 0.3, 0.6]:
                            obs = np.array(
                                [s0, s1, s2, s3, s4], dtype=np.float32
                            )
                            action = GAOptimizer._policy_from_gene(best_gene, obs)
                            state_idx = self._discretise(obs)
                            # Boost the GA-preferred action
                            self.q_table[state_idx, action] += bias_value
                            seeded_states += 1

        logger.info(
            "Phase 1 complete: fitness=%.1f, gene=%s, seeded %d Q-states "
            "(bias=%.2f) in %.1fs",
            self._ga_fitness,
            np.array2string(best_gene, precision=3),
            seeded_states,
            bias_value,
            self._ga_elapsed,
        )

        return {
            "best_gene": best_gene.tolist(),
            "best_fitness": float(self._ga_fitness),
            "generations": len(history),
            "seeded_states": seeded_states,
            "bias_value": float(bias_value),
            "elapsed_seconds": self._ga_elapsed,
        }

    # -- Phase 2: Q-Learning online control --------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection from the Q-table.

        Parameters
        ----------
        obs : ndarray, shape (5,)
            Normalised observation from InventoryEnv.

        Returns
        -------
        int
            Chosen action in {0, ..., n_actions-1}.
        """
        state_idx = self._discretise(obs)

        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            q_values = self.q_table[state_idx]
            # Break ties randomly
            max_q = np.max(q_values)
            candidates = np.where(np.isclose(q_values, max_q))[0]
            action = int(self.rng.choice(candidates))

        return action

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Perform a single Q-Learning (off-policy TD(0)) update.

        Parameters
        ----------
        obs : ndarray
            State before the action.
        action : int
            Action taken.
        reward : float
            Immediate reward received.
        next_obs : ndarray
            State after the action.
        done : bool
            Whether the episode terminated.
        """
        s = self._discretise(obs)
        s_next = self._discretise(next_obs)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[s_next])

        self.q_table[s, action] += self.alpha * (target - self.q_table[s, action])

        self._rl_step_count += 1
        self._current_episode_reward += reward

        if done:
            self._rl_episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
            self._rl_episode_count += 1
            # Decay epsilon at the end of every episode
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay
            )

    # -- episode boundary helper ------------------------------------------

    def on_episode_start(self) -> None:
        """Call at the beginning of each RL episode to reset accumulators."""
        self._current_episode_reward = 0.0

    # -- statistics -------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return combined GA and RL metrics.

        Returns
        -------
        dict
            Keys include GA results, RL training progress, Q-table statistics,
            and current hyper-parameter values.
        """
        q_nonzero = int(np.count_nonzero(self.q_table))
        q_total = int(self.q_table.size)

        recent_rewards = self._rl_episode_rewards[-100:]

        stats: dict[str, Any] = {
            # GA phase
            "ga": {
                "best_gene": (
                    self._ga_gene.tolist() if self._ga_gene is not None else None
                ),
                "best_fitness": float(self._ga_fitness),
                "generations_run": len(self._ga_history),
                "elapsed_seconds": self._ga_elapsed,
                "fitness_history": [
                    h["best_fitness"] for h in self._ga_history
                ],
            },
            # RL phase
            "rl": {
                "episodes": self._rl_episode_count,
                "total_steps": self._rl_step_count,
                "epsilon": float(self.epsilon),
                "alpha": float(self.alpha),
                "gamma": float(self.gamma),
                "mean_reward_last_100": (
                    float(np.mean(recent_rewards)) if recent_rewards else 0.0
                ),
                "best_episode_reward": (
                    float(np.max(self._rl_episode_rewards))
                    if self._rl_episode_rewards
                    else 0.0
                ),
                "episode_rewards": list(self._rl_episode_rewards),
            },
            # Q-table
            "q_table": {
                "shape": list(self.q_table.shape),
                "nonzero_entries": q_nonzero,
                "total_entries": q_total,
                "coverage_pct": round(100.0 * q_nonzero / max(q_total, 1), 2),
                "mean_q": float(np.mean(self.q_table)),
                "max_q": float(np.max(self.q_table)),
                "min_q": float(np.min(self.q_table)),
            },
        }

        return stats

    # -- convenience: full training loop -----------------------------------

    def train(
        self,
        env: Any,
        n_episodes: int = 500,
        run_ga: bool = True,
        log_every: int = 50,
    ) -> dict[str, Any]:
        """End-to-end training: optional GA warm-start then Q-Learning.

        Parameters
        ----------
        env : InventoryEnv
        n_episodes : int
            Number of RL episodes to run in Phase 2.
        run_ga : bool
            If ``True``, execute Phase 1 (GA) before RL training.
        log_every : int
            Log progress every *log_every* RL episodes.

        Returns
        -------
        dict
            Combined stats from both phases (see :meth:`get_stats`).
        """
        # Phase 1
        if run_ga:
            self.initialize_from_ga(env)
        else:
            logger.info("Skipping GA phase (run_ga=False).")

        # Phase 2
        logger.info(
            "Phase 2: Q-Learning for %d episodes (epsilon=%.3f) ...",
            n_episodes,
            self.epsilon,
        )
        t0 = time.perf_counter()

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            self.on_episode_start()
            done = False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs

            if ep % log_every == 0 or ep == n_episodes:
                recent = self._rl_episode_rewards[-log_every:]
                mean_r = float(np.mean(recent)) if recent else 0.0
                logger.info(
                    "RL ep %4d/%d  mean_reward(last %d)=%.1f  eps=%.4f",
                    ep,
                    n_episodes,
                    log_every,
                    mean_r,
                    self.epsilon,
                )

        elapsed_rl = time.perf_counter() - t0
        logger.info("Phase 2 complete in %.1fs.", elapsed_rl)

        return self.get_stats()

    def __repr__(self) -> str:
        return (
            f"HybridGARLAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"ga_pop={self.ga_pop}, ga_gens={self.ga_gens}, "
            f"epsilon={self.epsilon:.3f}, episodes={self._rl_episode_count})"
        )
