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
        max_active_reports: int = 2000,
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

        # internal state
        self.current_step = 0
        self.episode_violations_fixed = 0
        self.violations: List[ViolationReport] = []
        self.inspectors_available: List[bool] = []

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
        self.action_space = spaces.MultiDiscrete(
            [self.max_active_reports + 1] * (self.num_inspectors * self.inspection_rate)
        )

    # ------------------------------------------------------------------
    # environment life‑cycle methods
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode by sampling ``num_reports`` observations from the
        dataset and resetting all bookkeeping state.
        """
        # initialise chronological pointers and state
        self.current_step = 0
        self.episode_violations_fixed = 0
        self.inspectors_available = [True] * self.num_inspectors

        # chronological pointers
        if len(self._df_master) == 0:
            raise RuntimeError("No data available in _df_master to run environment")

        self._next_idx = 0
        first_date = self._df_master.loc[0, "Created Date"].date()
        self.current_date = first_date
        self.end_date = first_date + datetime.timedelta(days=self.time_steps - 1)

        # active pool of ViolationReport objects (chronological admission)
        self.violations = []

        # admit cases for the first date so the initial observation includes them
        self._admit_reports_for_date(self.current_date)

        return self._get_observation()

    def step(self, action: np.ndarray):
        """Apply the inspectors' choices for this day and return the new state,
        reward, done flag and info dict."""
        # Process actions for the current day
        violations_inspected = set()
        inspection_details = []  # record per-report features/outcomes
        reward = 0.0

        for insp_id in range(self.num_inspectors):
            for slot in range(self.inspection_rate):
                pos = insp_id * self.inspection_rate + slot
                choice = int(action[pos])
                # 0 = skip, otherwise 1-based index into active slots
                if choice == 0:
                    continue
                if choice > len(self.violations):
                    # chosen an inactive/padded slot
                    continue

                vidx = choice - 1
                if vidx < 0 or vidx >= len(self.violations):
                    continue

                violations_inspected.add(vidx)
                violation = self.violations[vidx]
                if violation.resolved:
                    continue
                # record pre-inspection state
                inspection_details.append({
                    'report_idx': vidx,
                    'days_outstanding_before': violation.days_outstanding,
                    'address': violation.address,
                    'borough': violation.borough,
                })

                # sample outcome
                roll = np.random.random()
                cum = 0.0
                for res, prob in self.outcome_probs.items():
                    cum += prob
                    if roll < cum:
                        violation.outcome = res
                        break
                violation.resolved = violation.outcome != Resolution.OPEN
                # append outcome to the inspection record
                inspection_details[-1]['outcome'] = violation.outcome.name

                # accumulate reward for this individual inspection
                if violation.outcome == Resolution.VIOLATION:
                    reward += 20.0
                    self.episode_violations_fixed += 1
                elif violation.outcome == Resolution.DUPLICATE:
                    reward += 3.0
                elif violation.outcome in (Resolution.NO_ACCESS, Resolution.CORRECTED, Resolution.NO_VIOLATION):
                    reward += 1.0

        # step penalty for open reports (encourage throughput)
        open_count = sum(1 for v in self.violations if not v.resolved)
        reward += -0.1 * open_count

        # age unresolved reports by 1 day
        for v in self.violations:
            if not v.resolved:
                v.days_outstanding += 1.0

        # advance the simulation date and admit new reports for the next day
        self.current_date = self.current_date + datetime.timedelta(days=1)
        self._admit_reports_for_date(self.current_date)

        self.current_step += 1

        done = (self.current_step >= self.time_steps) or (self.current_date > self.end_date)

        obs = self._get_observation()
        info = {
            "open_reports": open_count,
            "inspections_this_step": len(violations_inspected),
            "total_resolved": self.episode_violations_fixed,
            "current_date": str(self.current_date),
            "inspection_details": inspection_details,
        }
        return obs, reward, done, info

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
