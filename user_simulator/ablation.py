"""Simulator-component controls from Mind2Dialogue Table 8."""

from dataclasses import dataclass


@dataclass
class AblationConfig:
    """Toggle state tracking and behavior control for the paper's ablations.

    full: state and behavior controller.
    no_behavior: state without the behavior controller.
    no_state: neither state nor behavior controller (vanilla).
    oracle_profile_only: behavior controller without state, profile-only Oracle.
    """

    use_user_state: bool = True
    use_behavior_injection: bool = True
    history_window: int | None = None

    assistant_strategy: str = "oracle"

    sft_include_profile: bool = True

    name: str = "full"

    user_temperature: float = 0.7
    user_max_tokens: int = 2048
    user_retry_temps: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1)

    assistant_temperature: float = 0.7
    assistant_max_tokens: int = 1024

    controller_temperature: float = 0.9
    controller_max_tokens: int = 128

    scenario_constructor_temperature: float = 0.8
    scenario_constructor_max_tokens: int = 4096

    recent_history_window: int = 4

    @classmethod
    def full(cls) -> "AblationConfig":
        """Full stateful simulator and behavior controller with the Oracle."""
        return cls(name="full")


    @classmethod
    def no_behavior(cls) -> "AblationConfig":
        """Stateful user without the behavior controller, paired with the Oracle."""
        return cls(use_behavior_injection=False, name="no_behavior")

    @classmethod
    def no_state(cls) -> "AblationConfig":
        """Vanilla user without state or behavior control; assistant receives profile."""
        return cls(use_user_state=False, use_behavior_injection=False, name="no_state")


    @classmethod
    def oracle_profile_only(cls) -> "AblationConfig":
        """Behavior-controlled user without state tracking; profile-only Oracle."""
        return cls(use_user_state=False, assistant_strategy="oracle_profile_only", name="oracle_profile_only")

    @classmethod
    def from_name(cls, name: str) -> "AblationConfig":
        presets = {
            "full": cls.full,
            "no_behavior": cls.no_behavior,
            "no_state": cls.no_state,
            "oracle_profile_only": cls.oracle_profile_only,
        }
        factory = presets.get(name)
        if factory is None:
            raise ValueError(f"Unknown ablation: {name!r}. Choose from {list(presets)}")
        return factory()
