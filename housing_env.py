"""
Custom OpenAI Gym Environment for Housing Code Violation Inspection Optimization

This environment models the problem of allocating housing inspectors to code violation
reports from NYC's 311 service to maximize the number of violations fixed.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class InspectionAction(Enum):
    """Enumeration of possible inspection case outcomes."""
    NO_ACTION = 0
    ISSUE_WARNING = 1
    CLOSE_DUPLICATE = 2
    CLOSE_RESOLVED = 3


@dataclass
class ViolationReport:
    """Represents a 311 housing code violation report."""
    report_id: int
    no_heat: bool # Whether report is for no heat
    no_hot_water: bool # Whether report is for no hot water
    address: str  # Address
    street_name: str # street name
    borough: str # Borough
    council_district: int # Council District
    building_wide: bool # Whether report is building-wide
    date_reported: str  # Date of report
    days_outstanding: int  # Days since report was filed
    priority_score: float  # Calculated priority (0-1)
    resolved_date: str # date that report was resolved
    resolved: bool = False  # Whether report has been resolved -- either duplicate or fixed
    duplicate: bool = False # Whether report is a duplicate for a building-wide issue
    status: str = "OPEN"  # Status: OPEN, WARNING_ISSUED, DUPLICATE, RESOLVED
    inspection_count: int = 0  # Number of times this report has been inspected

@dataclass
class Inspector:
    """Represents an available inspector."""
    inspector_id: int
    available: bool = True
    violations_fixed_count: int = 0

class HousingViolationEnv(gym.Env):
    """
    Custom Gym environment for optimizing housing code violation inspections.
    
    State Space:
    - Active violation reports with their attributes
    - Inspector availability
    - Historical inspection outcomes
    
    Action Space:
    - Allocate each available inspector to a specific violation report
    - Pass (do not allocate to a report)
    
    Reward:
    - Positive reward for violations marked as resolved
    - Positive reward for violations inspected and marked as duplicate
    - Positive reward for violations inspected and warning issued
    - Penalty for violations remaining uninspected
    - Penalty for violations remaining open despite warning
    """
    
    def __init__(
        self,
        num_inspectors: int = 100,
        num_reports: int = 50,
        inspection_rate: int = 4,
        time_steps: int = 365,
        violation_data_path: Optional[str] = None,
    ):
        """
        Initialize the Housing Violation Inspection environment.
        
        Args:
            num_inspectors: Number of available inspectors
            num_reports: Number of reports read in per timestep
            inspection_rate: Number of inspections an individual inspector can do per day
            time_steps: Number of timesteps in an episode (day)
            violation_data_path: Path to 311 CSV data file
        """
        super(HousingViolationEnv, self).__init__()
        
        # ====================
        # Configuration
        # ====================
        self.num_inspectors = num_inspectors
        self.num_reports = num_reports
        self.time_steps = time_steps
        self.inspection_rate = inspection_rate
        self.violation_data_path = violation_data_path
        
        # Inspection outcome probabilities
        self.inspection_outcome_probs = {
            'issue_warning': 0.5,
            'close_duplicate': 0.1,
            'close_resolved': 0.1
        }
        
        # ====================
        # STATE SPACE CONFIGURATION
        # ====================
        # Each violation report is represented by:
        # - days_outstanding (continuous, 0-365)
        # - priority_score (continuous, 0-1)
        # - address (discrete)
        # - street_name (discrete)
        # - borough (discrete, 0-4)
        # - council_district (discrete, 0-50)
        # - building_wide (discrete, 0-1)
        # - date_reported (discrete)
        # - resolved_date (discrete)
        # - resolved (discrete, 0-1)
        # - duplicate (discrete, 0-1)

        # Total: 11 features per report
        num_reports = 50
        violation_features_per_report = 11
        
        # Inspector state features:
        # - available (binary, 0 or 1) -> num_inspectors
        # Total: 1 feature per inspector
        
        inspector_features = num_inspectors * 1
        
        # Total state size
        total_state_size = num_reports*violation_features_per_report + inspector_features + 1  # +1 for timestep
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_state_size,),
            dtype=np.float32
        )
        
        # ====================
        # ACTION SPACE CONFIGURATION
        # ====================
        # Each inspector can perform up to 4 inspections per timestep
        # For each inspection, they choose which violation report to inspect:
        # - 0 = no action (skip this inspection slot)
        # - 1 to num_reports = report index to inspect
        #
        # Case outcomes are stochastically determined:
        # - P(issue_warning) = 0.5
        # - P(close_as_duplicate) = 0.1  
        # - P(close_as_resolved) = 0.1
        # - P(remaining_open) = 0.3
        #
        # Action space: MultiDiscrete with 4 slots per inspector
        # Total length: num_inspectors * 4
        self.action_space = spaces.MultiDiscrete(
            [num_reports + 1] * (num_inspectors * 4)  # 4 inspections per inspector
        )
        
        # ====================
        # Environment State Initialization
        # ====================
        self.violations: List[ViolationReport] = []
        self.inspectors: List[Inspector] = []
        self.current_step = 0
        self.episode_violations_fixed = 0
        
        self._initialize_violations()
        self._initialize_inspectors()
    
    def _initialize_violations(self):
        """Initialize violation reports from data or synthetic generation."""
        self.violations = []
        
        if self.violation_data_path:
            self._load_violations_from_csv(self.violation_data_path)
        else:
            self._generate_synthetic_violations()
    
    def _load_violations_from_csv(self, csv_path: str):
        """
        Load violation reports from 311 CSV data.
        
        Expected CSV columns:
        - Unique ID (or index)
        - Location (address)
        - Complaint Type (violation type)
        - Complaint Date
        - Status (to help identify fixed violations)
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Filter to housing-related complaints if possible
            # Customize these filters based on your actual data structure
            housing_keywords = ['housing', 'heat', 'water', 'plumbing', 'electric', 'paint']
            # df = df[df['Complaint Type'].str.lower().str.contains('|'.join(housing_keywords))]
            
            for idx, row in df.iloc[:self.num_reports].iterrows():
                violation = ViolationReport(
                    report_id=idx,
                    location=str(row.get('Location', 'UNKNOWN')),
                    violation_type=str(row.get('Complaint Type', 'UNKNOWN')),
                    date_reported=str(row.get('Created Date', 'UNKNOWN')),
                    days_outstanding=max(0, np.random.randint(0, 365)),
                    priority_score=0.5  # TOOD - Will be calculated
                )
                violation.priority_score = self._calculate_priority(violation)
                self.violations.append(violation)
            
            print(f"Loaded {len(self.violations)} violation reports from {csv_path}")
        
        except FileNotFoundError:
            print(f"CSV file not found at {csv_path}. Using synthetic data instead.")
            self._generate_synthetic_violations()
        except ImportError:
            print("pandas not installed. Using synthetic data instead.")
            self._generate_synthetic_violations()
    
    def _generate_synthetic_violations(self):
        """Generate synthetic violation reports for testing."""
        violation_types = [
            'Heat/Hot Water', 'Plumbing', 'Paint/Wall Condition', 
            'Electric', 'Water Leak', 'Mold', 'Rodent', 
            'Ceiling', 'Floor', 'Window/Door'
        ]
        
        for i in range(self.num_reports):
            violation = ViolationReport(
                report_id=i,
                location=f"Address_{np.random.randint(1000, 99999)}",
                violation_type=np.random.choice(violation_types),
                date_reported=f"2025-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
                days_outstanding=np.random.randint(1, 365),
                priority_score=0.5
            )
            violation.priority_score = self._calculate_priority(violation)
            self.violations.append(violation)
    
    def _calculate_priority(self, violation: ViolationReport) -> float:
        """
        Calculate priority score for a violation.
        
        Priority factors:
        - Severity (higher = higher priority)
        - Days outstanding (longer = higher priority)
        
        Customize this based on your specific requirements.
        """
        severity_weight = 0.6
        age_weight = 0.4
        
        # Normalize days_outstanding to 0-1
        max_days = 365
        age_score = min(violation.days_outstanding / max_days, 1.0)
        
        priority = (severity_weight * violation.severity) + (age_weight * age_score)
        return min(priority, 1.0)
    
    def _initialize_inspectors(self):
        """Initialize inspector availability."""
        self.inspectors = [
            Inspector(inspector_id=i)
            for i in range(self.num_inspectors)
        ]
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation (state)
        """
        self.current_step = 0
        self.episode_violations_fixed = 0
        
        # Reset violations
        for violation in self.violations:
            violation.fixed = False
            violation.resolved = False
            violation.duplicate = False
            violation.status = "OPEN"
            violation.inspection_count = 0
        
        # Reset inspectors
        self._initialize_inspectors()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Array of length (num_inspectors * 4) representing up to 4 
                    inspection assignments per inspector.
                    Each element in [0, num_reports]:
                    0 = no action (skip this inspection slot)
                    1 to num_reports = report index to inspect
        
        Returns:
            observation: Updated state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # ====================
        # PROCESS ACTIONS
        # ====================
        # Track inspection outcomes
        violations_warned = 0
        violations_duplicate = 0
        violations_resolved = 0
        violations_inspected = []
        
        # Process each inspector's actions (up to 4 inspections per inspector)
        inspections_per_inspector = 4
        for inspector_id in range(self.num_inspectors):
            inspection_count = 0
            
            # Process each of the 4 inspection slots for this inspector
            for slot in range(inspections_per_inspector):
                action_idx_pos = inspector_id * inspections_per_inspector + slot
                action_idx = int(action[action_idx_pos])
                
                # Validate action
                if action_idx == 0 or action_idx > len(self.violations):
                    # No action or invalid action
                    continue
                
                # Check if inspector has already done 4 inspections
                if inspection_count >= inspections_per_inspector:
                    break
                
                violation_idx = action_idx - 1
                violation = self.violations[violation_idx]
                
                # Skip if already resolved
                if violation.resolved:
                    continue
                
                # Record inspection
                inspection_count += 1
                violations_inspected.append(violation_idx)
                violation.inspection_count += 1
                
                # Determine case outcome stochastically
                outcome_roll = np.random.random()
                cumulative_prob = 0.0
                
                if outcome_roll < self.inspection_outcome_probs['issue_warning']:
                    # Issue warning (50%)
                    violation.status = "WARNING_ISSUED"
                    violations_warned += 1
                elif outcome_roll < (self.inspection_outcome_probs['issue_warning'] + 
                                    self.inspection_outcome_probs['close_duplicate']):
                    # Close as duplicate (10%)
                    violation.status = "DUPLICATE"
                    violation.resolved = True
                    violation.duplicate = True
                    violations_duplicate += 1
                elif outcome_roll < (self.inspection_outcome_probs['issue_warning'] + 
                                    self.inspection_outcome_probs['close_duplicate'] +
                                    self.inspection_outcome_probs['close_resolved']):
                    # Close as resolved (10%)
                    violation.status = "RESOLVED"
                    violation.resolved = True
                    violations_resolved += 1
                    self.episode_violations_fixed += 1
                else:
                    # Remains open (30%)
                    violation.status = "OPEN"
        
        # ====================
        # CALCULATE REWARD
        # ====================
        reward = self._calculate_reward(
            violations_resolved,
            violations_duplicate,
            violations_warned,
            len(set(violations_inspected))
        )
        
        # ====================
        # CHECK TERMINATION
        # ====================
        done = self.current_step >= self.time_steps
        
        # ====================
        # GET INFO
        # ====================
        info = {
            'violations_resolved': violations_resolved,
            'violations_marked_duplicate': violations_duplicate,
            'violations_warned': violations_warned,
            'total_inspections': len(violations_inspected),
            'unique_violations_inspected': len(set(violations_inspected)),
            'timestep': self.current_step,
            'total_resolved_to_date': self.episode_violations_fixed
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _calculate_reward(self, resolved: int, duplicate: int, warned: int, total_inspected: int) -> float:
        """
        Calculate reward for this step.
        
        Reward components:
        - Resolved violations: +20 per violation
        - Duplicate violations: +10 per violation
        - Warnings issued: +2 per warning
        - Penalty for uninspected open violations
        
        Customize this based on your objectives.
        """
        # Reward for fixing violations
        resolve_reward = resolved * 20.0
        duplicate_reward = duplicate * 10.0
        warning_reward = warned * 2.0
        
        # Penalty for violations still open
        open_violations = sum(1 for v in self.violations if v.status == "OPEN")
        open_penalty = open_violations * -0.5
        
        return resolve_reward + duplicate_reward + warning_reward + open_penalty
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the current observation (state) vector.
        
        State encoding:
        - For each violation: [violation_type_encoded, severity, days_outstanding, priority_score]
        - For each inspector: [available, utilization]
        - Current timestep (normalized)
        
        Returns:
            Normalized state vector as float32 array
        """
        state = []
        
        # Add violation features
        for violation in self.violations:
            # Encode violation type (0-9 for 10 types, normalized to 0-1)
            violation_types = [
                'Heat/Hot Water', 'Plumbing', 'Paint/Wall Condition', 
                'Electric', 'Water Leak', 'Mold', 'Rodent', 
                'Ceiling', 'Floor', 'Window/Door'
            ]
            
            try:
                type_idx = violation_types.index(violation.violation_type)
            except ValueError:
                type_idx = 0
            
            type_encoded = type_idx / 10.0  # Normalize to 0-1
            
            state.extend([
                type_encoded,
                violation.severity,
                min(violation.days_outstanding / 365.0, 1.0),  # Normalize to 0-1
                violation.priority_score
            ])
        
        # Add inspector features
        for inspector in self.inspectors:
            state.extend([
                float(inspector.available),
                inspector.utilization
            ])
        
        # Add timestep (normalized)
        state.append(self.current_step / self.time_steps)
        
        return np.array(state, dtype=np.float32)
    
    def render(self, mode: str = 'human'):
        """
        Render the environment state.
        
        Args:
            mode: 'human' for console output
        """
        if mode == 'human':
            print(f"\n--- Step {self.current_step}/{self.time_steps} ---")
            print(f"Violations Fixed This Episode: {self.episode_violations_fixed}")
            print(f"Active Inspectors: {sum(1 for i in self.inspectors if i.available)}")
            
            # Show top 5 violations by priority
            sorted_violations = sorted(
                self.violations, 
                key=lambda v: v.priority_score, 
                reverse=True
            )
            
            print("\nTop 5 Priority Violations:")
            for v in sorted_violations[:5]:
                status = "FIXED" if v.fixed else "OPEN"
                print(
                    f"  [{status}] {v.violation_type} | Priority: {v.priority_score:.2f} | "
                    f"Days Outstanding: {v.days_outstanding}"
                )
    
    def close(self):
        """Clean up environment resources."""
        pass


# ====================
# EXAMPLE USAGE
# ====================
if __name__ == "__main__":
    # Create environment with synthetic data
    env = HousingViolationEnv(
        num_inspectors=100,
        num_reports=50,
        time_steps=100
    )
    
    # Or load from 311 CSV:
    # env = HousingViolationEnv(
    #     num_inspectors=10,
    #     num_reports=100,
    #     time_steps=365,
    #     violation_data_path="data/311_Service_Requests_from_2020_to_Present_20260224.csv"
    # )
    
    # Example episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(5):
        # Random action: allocate each inspector randomly
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Info: {info}")
        
        env.render()
        
        if done:
            break
    
    env.close()
