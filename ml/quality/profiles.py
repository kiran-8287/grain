"""
Whole-grain reference profile management.

Defines and validates reference profiles for whole-grain dimensions (length, breadth, L/B ratio).
Profiles can be defined in pixels (for fixed imaging setups) or calibrated millimeters.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ml.config import get_project_root

logger = logging.getLogger(__name__)


@dataclass
class GrainProfile:
    """Reference profile describing the expected intact kernel dimensions."""
    profile_name: str
    reference_unit: str  # "pixels" or "mm"
    whole_kernel_length: float
    whole_kernel_breadth: Optional[float] = None
    whole_kernel_lb_ratio: Optional[float] = None
    source: str = "configured_profile"
    reference_count: int = 1
    notes: Optional[str] = None

    def validate(self) -> Tuple_Validation:
        """Validate that the profile contains usable, non-corrupt measurements."""
        if not self.profile_name:
            return False, "Profile name is empty"
        if self.reference_unit not in ("pixels", "mm"):
            return False, f"Invalid reference unit '{self.reference_unit}', must be 'pixels' or 'mm'"
        if self.whole_kernel_length <= 0:
            return False, f"Invalid whole_kernel_length {self.whole_kernel_length}, must be > 0"
        if self.whole_kernel_breadth is not None and self.whole_kernel_breadth <= 0:
            return False, f"Invalid whole_kernel_breadth {self.whole_kernel_breadth}, must be > 0"
        if self.whole_kernel_lb_ratio is not None and self.whole_kernel_lb_ratio <= 0:
            return False, f"Invalid whole_kernel_lb_ratio {self.whole_kernel_lb_ratio}, must be > 0"
        return True, "Valid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Tuple_Validation = tuple[bool, str]


def load_grain_profile(profile_path_or_name: Union[str, Path, Dict[str, Any]]) -> Optional[GrainProfile]:
    """
    Load and validate a GrainProfile from a dict, file path, or profile name.
    
    Searches:
    1. Direct file path if exists
    2. grain_profiles/<name>.json (or <name> if has .json)
    3. configs/grain_profiles/<name>.json
    """
    if isinstance(profile_path_or_name, dict):
        try:
            profile = GrainProfile(
                profile_name=str(profile_path_or_name.get("profile_name", "custom")),
                reference_unit=str(profile_path_or_name.get("reference_unit", "pixels")),
                whole_kernel_length=float(profile_path_or_name.get("whole_kernel_length", 0.0)),
                whole_kernel_breadth=(
                    float(profile_path_or_name["whole_kernel_breadth"])
                    if profile_path_or_name.get("whole_kernel_breadth") is not None
                    else None
                ),
                whole_kernel_lb_ratio=(
                    float(profile_path_or_name["whole_kernel_lb_ratio"])
                    if profile_path_or_name.get("whole_kernel_lb_ratio") is not None
                    else None
                ),
                source=str(profile_path_or_name.get("source", "dict_input")),
                reference_count=int(profile_path_or_name.get("reference_count", 1)),
                notes=profile_path_or_name.get("notes"),
            )
            is_valid, msg = profile.validate()
            if not is_valid:
                logger.warning(f"Invalid grain profile dict: {msg}")
                return None
            return profile
        except Exception as exc:
            logger.warning(f"Failed to parse grain profile dict: {exc}")
            return None

    project_root = get_project_root()
    path_candidate = Path(profile_path_or_name)

    candidates = [
        path_candidate,
        project_root / path_candidate,
        project_root / "grain_profiles" / path_candidate,
        project_root / "grain_profiles" / f"{path_candidate.stem}.json",
        project_root / "configs" / "grain_profiles" / f"{path_candidate.stem}.json",
    ]

    target_file: Optional[Path] = None
    for cand in candidates:
        if cand.is_file():
            target_file = cand
            break

    if target_file is None:
        logger.debug(f"Grain profile not found for '{profile_path_or_name}'")
        return None

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = GrainProfile(
            profile_name=str(data.get("profile_name", target_file.stem)),
            reference_unit=str(data.get("reference_unit", "pixels")),
            whole_kernel_length=float(data.get("whole_kernel_length", 0.0)),
            whole_kernel_breadth=(
                float(data["whole_kernel_breadth"])
                if data.get("whole_kernel_breadth") is not None
                else None
            ),
            whole_kernel_lb_ratio=(
                float(data["whole_kernel_lb_ratio"])
                if data.get("whole_kernel_lb_ratio") is not None
                else None
            ),
            source=str(data.get("source", f"file:{target_file.name}")),
            reference_count=int(data.get("reference_count", 1)),
            notes=data.get("notes"),
        )
        is_valid, msg = profile.validate()
        if not is_valid:
            logger.warning(f"Grain profile in {target_file} is invalid: {msg}")
            return None
        return profile
    except Exception as exc:
        logger.warning(f"Failed to read grain profile from {target_file}: {exc}")
        return None


def get_default_grain_profile() -> Optional[GrainProfile]:
    """Look for default_rice profile in project roots."""
    return load_grain_profile("default_rice")
