"""
Revised OpenAI Gym Environment for Housing Code Violation Inspection Optimization

This version is tailored to the preprocessed 311 data in
`data/311_preproc.csv`.  It reflects the problem formulation described by the
user:

- five resolution categories (no access, duplicate, corrected, no violation,
  violation) that account for the vast majority of heat/hot water complaints,
  and all other reports have been discarded during preprocessing.
- inspectors make up to four inspections per day and select which reports to
  visit; the environment stochastically determines the outcome using
  empirically-derived probabilities.
- positive rewards are given for closing reports, with the largest incentive
  allocated to issuing violations; duplicates are worth a small bonus because
  they require no travel; cases that consume travel resources but close
  without a violation earn a modest reward.  There is a per‑timestep penalty for
  each open report to encourage overall throughput.

Geographic and problem attributes from the preprocessed dataset are encoded as
numeric features in the state vector so that learning agents may condition on
them.  The environment does _not_ currently enforce borough‑specific inspector
constraints, but those could be layered on later.

To initialise the environment give the path to your preprocessed CSV file;
otherwise the environment will raise an error when reset() is called.
"""

import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class Resolution(Enum):
    OPEN = 0
    NO_ACCESS = 1
    DUPLICATE = 2
    CORRECTED = 3
    NO_VIOLATION = 4
    VIOLATION = 5


@dataclass
class RewardWeights:
    """Controls the relative importance of each reward component.

    The base reward for an inspection is decomposed as:
        accuracy_bonus  = _ACCURACY_BONUS  if outcome == VIOLATION else 0
        throughput_base = _THROUGHPUT_REWARD[outcome]
        fairness_bonus  = (expected_borough_share - actual_borough_share) * _FAIRNESS_SCALE

        step_reward = w_accuracy * accuracy_bonus
                    + w_throughput * throughput_base
                    + w_fairness * fairness_bonus

        episode_penalty = -open_penalty * open_report_count  (added once per timestep)

    Default values reproduce the original hardcoded reward structure:
        VIOLATION    -> 40 * 1.0 + 10 * 1.0 = 50
        DUPLICATE    ->  0 * 1.0 + 20 * 1.0 = 20
        others       ->  0 * 1.0 + 10 * 1.0 = 10
    """
    w_accuracy: float = 1.0   # weight on the violation-detection bonus (40 pts base)
    w_throughput: float = 1.0 # weight on the report-closure reward (10–20 pts base)
    w_fairness: float = 0.0   # weight on the borough equity bonus (±10 pts base)
    open_penalty: float = 0.0 # per-open-report penalty applied once per timestep


# Base reward constants (scaled by RewardWeights at runtime)
_ACCURACY_BONUS = 40.0          # extra reward exclusively for finding a real violation
_THROUGHPUT_REWARD: Dict[Resolution, float] = {
    Resolution.VIOLATION:    10.0,
    Resolution.DUPLICATE:    20.0,
    Resolution.NO_ACCESS:    10.0,
    Resolution.CORRECTED:    10.0,
    Resolution.NO_VIOLATION: 10.0,
}
_FAIRNESS_SCALE = 10.0          # multiplier for (expected_share - actual_share)


@dataclass
class ViolationReport:
    """Represents a single 311 housing report extracted from the preprocessed
    file.  All of the attributes correspond directly to columns in
    ``311_preproc.csv``.
    """
    report_id: int
    address: str
    street_name: str
    borough: str
    council_district: float
    apartment_only: int
    building_wide: int
    no_heat: int
    no_hot_water: int
    created_date: pd.Timestamp
    days_outstanding: float

    # labels present in the dataset (not used for transition dynamics but handy
    # for evaluation/analysis).
    no_access: int
    duplicate: int
    corrected: int
    no_violation_issued: int
    violation_issued: int

    # state maintained by the environment during an episode
    resolved: bool = False
    outcome: Resolution = Resolution.OPEN


class HousingEnv(gym.Env):
    """Gym environment modelling the inspector allocation problem described
    by the user.  It makes no attempt to enforce geographic constraints or a
    variable workforce; each episode simply samples ``num_reports`` from the
    preprocessed dataset and keeps them fixed for the duration of the episode.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_inspectors: int = 100,
        num_reports: int = 1000,
        inspection_rate: int = 4,
        years: int = 4,
        data_path: Optional[str] = "data/311_preproc.csv",
           max_active_reports: int = 300,
           hierarchical: bool = True,
           start_date_str: Optional[str] = None,
           reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__()

        # configuration parameters
        self.num_inspectors = num_inspectors
        self.num_reports = num_reports
        self.inspection_rate = inspection_rate
        # episodes will span `years` years (approx); convert to days
        self.years = years
        self.time_steps = int(self.years * 365)
        self.data_path = data_path
        self.max_active_reports = max_active_reports
        self.hierarchical = hierarchical
        self.start_date_str = start_date_str
        self.reward_weights = reward_weights if reward_weights is not None else RewardWeights()

        # internal state
        self.current_step = 0
        self.episode_violations_fixed = 0
        self.violations: List[ViolationReport] = []
        self.inspectors_available: List[bool] = []
        # borough tracking for fairness reward
        self._borough_inspections: Dict[str, int] = {}
        self._borough_report_counts: Dict[str, int] = {}

        # load and prepare the dataset immediately so we can compute encodings
        self._load_and_prepare_data()

        # build encoders for categorical attributes that will appear in the
        # observation vector
        self._build_encoders()

        # outcome probabilities (sum to 1)
        self.outcome_probs: Dict[Resolution, float] = {
            Resolution.NO_ACCESS: 0.14,
            Resolution.DUPLICATE: 0.33,
            Resolution.CORRECTED: 0.33,
            Resolution.VIOLATION: 0.10,
            Resolution.NO_VIOLATION: 0.10,
        }

        # construct observation & action spaces based on the configuration
        self._configure_spaces()

    # ------------------------------------------------------------------
    # data loading / preprocessing
    # ------------------------------------------------------------------

    def _load_and_prepare_data(self):
        """Read CSV and filter to rows that contain one of the five core
        resolution types.  Compute ``days_outstanding`` from the provided
        duration column (already present during preprocessing) so that policies
        can condition on it directly.
        """
        if self.data_path is None:
            raise ValueError("data_path must be provided to load real data")

        df = pd.read_csv(self.data_path)
        # ensure necessary columns are present
        required_cols = [
            "Incident Address",
            "Street Name",
            "Borough",
            "Council District",
            "Apartment_Only",
            "Entire_Building",
            "No_Heat",
            "No_Hot_Water",
            "Created Date",
            "Duration_Days",
            "No_Access",
            "Duplicate",
            "Corrected",
            "No_Violation_Issued",
            "Violation_Issued",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Dataset missing expected column(s): {missing}")

        # filter out cases with none of the five core resolutions
        df = df[                                           # keep at least one flag
            df["No_Access"]
            + df["Duplicate"]
            + df["Corrected"]
            + df["No_Violation_Issued"]
            + df["Violation_Issued"]
            > 0
        ].copy()

        # parse dates and normalise the Created Date to date-only (floor to day)
        df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
        df = df.dropna(subset=["Created Date"])  # drop rows we can't date
        df["Created Date"] = df["Created Date"].dt.floor("D")

        # days outstanding already exists as Duration_Days; use it and clip to
        # a reasonable bound for normalization later
        df["days_outstanding"] = df["Duration_Days"].fillna(0).astype(float)

        # sort chronologically to enable sequential admission
        self._df_master = df.sort_values("Created Date").reset_index(drop=True)

    def _build_encoders(self):
        """Create lookup tables for categorical text fields so they can be
        turned into numerical features in the observation vector.
        """
        df = self._df_master
        self._address_map = {v: i for i, v in enumerate(df["Incident Address"].fillna("<UNK>").unique())}
        self._street_map = {v: i for i, v in enumerate(df["Street Name"].fillna("<UNK>").unique())}
        self._borough_map = {v: i for i, v in enumerate(df["Borough"].fillna("<UNK>").unique())}

        # we will normalise indices by the max+1 when producing a vector
        self._max_address_idx = max(self._address_map.values()) if self._address_map else 1
        self._max_street_idx = max(self._street_map.values()) if self._street_map else 1
        self._max_borough_idx = max(self._borough_map.values()) if self._borough_map else 1

    def _get_outcome_probs(self, violation: ViolationReport) -> Dict[Resolution, float]:
        """Calculate outcome probabilities based on report features.
        
        Implements Options 1 & 2:
        - Option 1: Building-wide reports have higher violation probability
        - Option 2: No heat/hot water complaints have higher violation probability
        
        Returns a normalized probability distribution that sums to 1.0
        """
        # Base probabilities for all outcomes (excluding VIOLATION which is calculated)
        base_probs = {
            Resolution.NO_ACCESS: 0.14,
            Resolution.DUPLICATE: 0.33,
            Resolution.CORRECTED: 0.33,
            Resolution.NO_VIOLATION: 0.10,
        }
        
        violation_prob = 0.10  # Base violation probability
        
        # Option 1: Building-wide reports are more serious
        if violation.building_wide:
            violation_prob += 0.15
            base_probs[Resolution.NO_VIOLATION] -= 0.08  # Less likely to be harmless
            base_probs[Resolution.NO_ACCESS] -= 0.07     # Better access to shared building areas
        
        # Option 2: No heat/water are urgent health hazards
        if violation.no_heat or violation.no_hot_water:
            violation_prob += 0.08
            base_probs[Resolution.NO_VIOLATION] -= 0.05  # Health hazard, not innocent
            base_probs[Resolution.NO_ACCESS] -= 0.03
        
        # Ensure probabilities don't go negative
        for key in base_probs:
            base_probs[key] = max(base_probs[key], 0.01)
        
        # Add violation probability
        base_probs[Resolution.VIOLATION] = violation_prob
        
        # Normalize to ensure sum = 1.0
        total = sum(base_probs.values())
        normalized_probs = {k: v / total for k, v in base_probs.items()}
        
        return normalized_probs

    # ------------------------------------------------------------------
    # observation / action space helpers
    # ------------------------------------------------------------------

    def _configure_spaces(self):
        # determine number of scalar features per report
        # features: [address_code, street_code, borough_code, council_district,
        # apartment_only, building_wide, no_heat, no_hot_water,
        # days_outstanding_norm, created_month_norm]
        self._features_per_report = 10

        # Observations will be a Dict so we can carry a reports tensor and a mask
        reports_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_active_reports, self._features_per_report),
            dtype=np.float32,
        )
        mask_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_active_reports,), dtype=np.float32)
        inspectors_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_inspectors,), dtype=np.float32)
        timestep_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "reports": reports_space,
            "mask": mask_space,
            "inspectors": inspectors_space,
            "timestep": timestep_space,
        })

        # Action space: indices into the active pool slots (1-based), 0 = skip
        # Hierarchical action space: (inspector_id, report_id) -> Discrete(num_inspectors * max_active_reports)
        if self.hierarchical:
            self.action_space = spaces.Discrete(self.num_inspectors * self.max_active_reports)
        else:
            # Original flat action space (not recommended; mainly for backwards compatibility)
            self.action_space = spaces.MultiDiscrete(
                [self.max_active_reports + 1] * (self.num_inspectors * self.inspection_rate)
            )

    # ------------------------------------------------------------------
    # reward helpers
    # ------------------------------------------------------------------

    def _update_borough_report_counts(self) -> None:
        """Recount open (unresolved) reports per borough from the active pool."""
        counts: Dict[str, int] = {}
        for v in self.violations:
            if not v.resolved:
                counts[v.borough] = counts.get(v.borough, 0) + 1
        self._borough_report_counts = counts

    def _compute_fairness_bonus(self, borough: str) -> float:
        """Return a fairness bonus for inspecting `borough`.

        Positive when the borough is under-inspected relative to its share of
        open reports; negative when over-inspected.  Bounded to ±_FAIRNESS_SCALE.
        """
        total_reports = sum(self._borough_report_counts.values())
        if total_reports == 0:
            return 0.0
        expected_share = self._borough_report_counts.get(borough, 0) / total_reports

        total_insp = sum(self._borough_inspections.values())
        if total_insp == 0:
            actual_share = 0.0
        else:
            actual_share = self._borough_inspections.get(borough, 0) / total_insp

        return (expected_share - actual_share) * _FAIRNESS_SCALE

    def _compute_inspection_reward(self, violation: ViolationReport) -> float:
        """Compute the weighted reward for a single resolved inspection."""
        w = self.reward_weights
        outcome = violation.outcome

        accuracy_bonus = _ACCURACY_BONUS if outcome == Resolution.VIOLATION else 0.0
        throughput_base = _THROUGHPUT_REWARD.get(outcome, 0.0)
        fairness_bonus = self._compute_fairness_bonus(violation.borough) if w.w_fairness != 0.0 else 0.0

        return (
            w.w_accuracy  * accuracy_bonus
            + w.w_throughput * throughput_base
            + w.w_fairness   * fairness_bonus
        )

    def borough_equity_score(self) -> float:
        """Total-variation distance between inspection shares and report shares.

        Returns a value in [0, 1]: 0 = perfectly equitable, 1 = maximally skewed.
        Useful for offline analysis and logging.
        """
        boroughs = set(self._borough_report_counts) | set(self._borough_inspections)
        if not boroughs:
            return 0.0
        total_reports = max(1, sum(self._borough_report_counts.values()))
        total_insp    = max(1, sum(self._borough_inspections.values()))
        tv = sum(
            abs(
                self._borough_report_counts.get(b, 0) / total_reports
                - self._borough_inspections.get(b, 0) / total_insp
            )
            for b in boroughs
        )
        return tv / 2.0  # normalise to [0, 1]

    # ------------------------------------------------------------------
    # environment life‑cycle methods
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """Start a new episode by sampling ``num_reports`` observations from the
        dataset and resetting all bookkeeping state.
        """
        # handle seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # initialise chronological pointers and state
        self.current_step = 0
        self.episode_violations_fixed = 0
        self.inspectors_available = [True] * self.num_inspectors
        self._borough_inspections = {}
        self._borough_report_counts = {}

        # chronological pointers
        if len(self._df_master) == 0:
            raise RuntimeError("No data available in _df_master to run environment")

        # Determine starting index based on start_date_str
        if self.start_date_str:
            start_date = pd.to_datetime(self.start_date_str).date()
            # Find first index where Created Date >= start_date
            matching_idx = self._df_master[self._df_master['Created Date'] >= pd.to_datetime(self.start_date_str)]
            if len(matching_idx) == 0:
                raise ValueError(f"No data found on or after {self.start_date_str}")
            self._next_idx = matching_idx.index[0]
            first_date = start_date
        else:
            self._next_idx = 0
            first_date = self._df_master.loc[0, "Created Date"].date()
        
        self.current_date = first_date
        self.end_date = first_date + datetime.timedelta(days=self.time_steps - 1)

        # active pool of ViolationReport objects (chronological admission)
        self.violations = []

        # admit cases for the first date so the initial observation includes them
        self._admit_reports_for_date(self.current_date)

        obs = self._get_observation()
        info = {"current_date": str(self.current_date)}
        return obs, info

    def step(self, action: np.ndarray):
        """Apply the inspectors' choices for this day and return the new state,
        reward, done flag and info dict."""
        # Refresh report-count denominator used by the fairness term
        self._update_borough_report_counts()

        violations_inspected = set()
        inspection_details = []  # record per-report features/outcomes
        reward = 0.0

        if self.hierarchical:
            # Hierarchical action: single Discrete value
            # Decode: inspector_id = action // max_active_reports, report_id = action % max_active_reports
            action_val = int(action)
            insp_id = action_val // self.max_active_reports
            report_id = action_val % self.max_active_reports

            # Validate inspector and report indices
            if insp_id < 0 or insp_id >= self.num_inspectors:
                insp_id = 0
            if report_id >= len(self.violations):
                report_id = len(self.violations) - 1 if len(self.violations) > 0 else -1

            # Perform inspection if valid
            if report_id >= 0 and report_id < len(self.violations):
                vidx = report_id
                violation = self.violations[vidx]
                if not violation.resolved:
                    violations_inspected.add(vidx)
                    inspection_details.append({
                        'report_idx': vidx,
                        'days_outstanding_before': violation.days_outstanding,
                        'address': violation.address,
                        'borough': violation.borough,
                    })

                    # sample outcome using feature-based probabilities
                    outcome_probs = self._get_outcome_probs(violation)
                    roll = np.random.random()
                    cum = 0.0
                    for res, prob in outcome_probs.items():
                        cum += prob
                        if roll < cum:
                            violation.outcome = res
                            break
                    violation.resolved = violation.outcome != Resolution.OPEN
                    inspection_details[-1]['outcome'] = violation.outcome.name

                    # compute fairness bonus using pre-inspection shares, then record
                    reward += self._compute_inspection_reward(violation)
                    b = violation.borough
                    self._borough_inspections[b] = self._borough_inspections.get(b, 0) + 1

                    if violation.outcome == Resolution.VIOLATION:
                        self.episode_violations_fixed += 1
        else:
            # Original flat action space (multiple actions per step)
            for insp_id in range(self.num_inspectors):
                for slot in range(self.inspection_rate):
                    pos = insp_id * self.inspection_rate + slot
                    choice = int(action[pos])
                    # 0 = skip, otherwise 1-based index into active slots
                    if choice == 0:
                        continue
                    if choice > len(self.violations):
                        continue

                    vidx = choice - 1
                    if vidx < 0 or vidx >= len(self.violations):
                        continue

                    violations_inspected.add(vidx)
                    violation = self.violations[vidx]
                    if violation.resolved:
                        continue
                    inspection_details.append({
                        'report_idx': vidx,
                        'days_outstanding_before': violation.days_outstanding,
                        'address': violation.address,
                        'borough': violation.borough,
                    })

                    # sample outcome using feature-based probabilities
                    outcome_probs = self._get_outcome_probs(violation)
                    roll = np.random.random()
                    cum = 0.0
                    for res, prob in outcome_probs.items():
                        cum += prob
                        if roll < cum:
                            violation.outcome = res
                            break
                    violation.resolved = violation.outcome != Resolution.OPEN
                    inspection_details[-1]['outcome'] = violation.outcome.name

                    # compute fairness bonus using pre-inspection shares, then record
                    reward += self._compute_inspection_reward(violation)
                    b = violation.borough
                    self._borough_inspections[b] = self._borough_inspections.get(b, 0) + 1

                    if violation.outcome == Resolution.VIOLATION:
                        self.episode_violations_fixed += 1

        # count open reports for metrics
        open_count = sum(1 for v in self.violations if not v.resolved)

        # per-timestep open-report penalty (default 0 → no change in behaviour)
        reward -= self.reward_weights.open_penalty * open_count

        # age unresolved reports by 1 day
        for v in self.violations:
            if not v.resolved:
                v.days_outstanding += 1.0

        # advance the simulation date and admit new reports for the next day
        self.current_date = self.current_date + datetime.timedelta(days=1)
        self._admit_reports_for_date(self.current_date)

        self.current_step += 1

        terminated = (self.current_step >= self.time_steps) or (self.current_date > self.end_date)
        truncated = False  # No truncation in this environment

        obs = self._get_observation()
        info = {
            "open_reports": open_count,
            "inspections_this_step": len(violations_inspected),
            "total_resolved": self.episode_violations_fixed,
            "current_date": str(self.current_date),
            "inspection_details": inspection_details,
            "borough_equity_score": self.borough_equity_score(),
            "borough_inspections": dict(self._borough_inspections),
        }
        return obs, reward, terminated, truncated, info

    def _make_violation(self, idx: int, row: pd.Series) -> ViolationReport:
        return ViolationReport(
            report_id=int(idx),
            address=row.get("Incident Address", ""),
            street_name=row.get("Street Name", ""),
            borough=row.get("Borough", ""),
            council_district=float(row.get("Council District", 0)),
            apartment_only=int(row.get("Apartment_Only", 0)),
            building_wide=int(row.get("Entire_Building", 0)),
            no_heat=int(row.get("No_Heat", 0)),
            no_hot_water=int(row.get("No_Hot_Water", 0)),
            created_date=pd.to_datetime(row["Created Date"]),
            days_outstanding=float(row.get("days_outstanding", 0.0)),
            no_access=int(row.get("No_Access", 0)),
            duplicate=int(row.get("Duplicate", 0)),
            corrected=int(row.get("Corrected", 0)),
            no_violation_issued=int(row.get("No_Violation_Issued", 0)),
            violation_issued=int(row.get("Violation_Issued", 0)),
        )

    def _admit_reports_for_date(self, date: datetime.date):
        """Append reports whose Created Date == date to the active pool.
        Keeps the pool size bounded by trimming oldest unresolved reports if
        necessary to respect `self.max_active_reports`.
        """
        df = self._df_master
        n = len(df)
        added = 0
        while self._next_idx < n and df.loc[self._next_idx, "Created Date"].date() == date:
            row = df.loc[self._next_idx]
            v = self._make_violation(int(self._next_idx), row)
            self.violations.append(v)
            self._next_idx += 1
            added += 1

        # enforce max active pool size by dropping oldest unresolved reports
        if len(self.violations) > self.max_active_reports:
            # keep the most recent `max_active_reports` items
            self.violations = self.violations[-self.max_active_reports :]

        return added

    def _encode_violation(self, v: ViolationReport) -> List[float]:
        """Turn a single report into a list of normalized floats for the
        observation vector."""
        addr_code = self._address_map.get(v.address, 0) / self._max_address_idx
        street_code = self._street_map.get(v.street_name, 0) / self._max_street_idx
        borough_code = self._borough_map.get(v.borough, 0) / self._max_borough_idx
        cd_norm = min(v.council_district / 51.0, 1.0)
        days_norm = min(v.days_outstanding / 365.0, 1.0)
        created_month = v.created_date.month / 12.0 if pd.notna(v.created_date) else 0.0

        return [
            addr_code,
            street_code,
            borough_code,
            cd_norm,
            float(v.apartment_only),
            float(v.building_wide),
            float(v.no_heat),
            float(v.no_hot_water),
            days_norm,
            created_month,
        ]

    def _get_observation(self) -> np.ndarray:
        """Return a dictionary observation with:
        - reports: tensor shaped (max_active_reports, features_per_report)
        - mask: binary vector indicating which slots are active
        - inspectors: availability vector
        - timestep: normalized progress through episode
        """
        # prepare reports tensor and mask
        reports = np.zeros((self.max_active_reports, self._features_per_report), dtype=np.float32)
        mask = np.zeros((self.max_active_reports,), dtype=np.float32)

        for i, v in enumerate(self.violations[: self.max_active_reports]):
            reports[i, :] = np.array(self._encode_violation(v), dtype=np.float32)
            mask[i] = 1.0

        inspectors = np.array([1.0 if a else 0.0 for a in self.inspectors_available], dtype=np.float32)
        timestep = np.array([float(self.current_step) / max(1, self.time_steps)], dtype=np.float32)

        return {"reports": reports, "mask": mask, "inspectors": inspectors, "timestep": timestep}

    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step {self.current_step}/{self.time_steps}, open reports = {sum(1 for v in self.violations if not v.resolved)}")

    def close(self):
        pass


# --------------------------------------------------
# simple usage example
# --------------------------------------------------
if __name__ == "__main__":
    # example run: one year of data with a smaller active pool for testing
    env = HousingEnv(num_inspectors=10, num_reports=50, inspection_rate=4, years=1, max_active_reports=200)
    obs = env.reset()
    print("initial observation keys:", list(obs.keys()))
    print("reports shape:", obs["reports"].shape)
    for t in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(t, reward, info)
        env.render()
        if done:
            break
