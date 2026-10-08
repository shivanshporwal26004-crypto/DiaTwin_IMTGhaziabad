"""Shared constants for the DiaTwin pipeline."""

STEP_MIN = 5                      # sensor resolution (minutes) - typical CGM cadence
STEPS_PER_DAY = 24 * 60 // STEP_MIN
HORIZON_MIN = 120                 # prediction horizon (minutes)
HORIZON_STEPS = HORIZON_MIN // STEP_MIN
SPIKE_MGDL = 180                  # hyperglycaemia threshold (upper bound of ADA time-in-range)
HYPO_MGDL = 70                    # lower bound of time-in-range
WARMUP_STEPS = STEPS_PER_DAY      # first 24 h of each series is used only as history

HORIZONS_MIN = (30, 60, 120)      # multi-horizon forecasts
CGM_FFILL_LIMIT = 6               # carry last CGM reading forward for up to 30 min of sensor gap
