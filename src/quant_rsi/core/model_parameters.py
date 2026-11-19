"""The fixed LightGBM configuration shared by research fits and the frozen forecast workflow.

Neutral (#159 D2): research-study, rolling research and the forecast comparison protocol
all import these values from here, so equal configurations stay equal by construction.
"""

MODEL_PARAMETERS = dict(
    objective="regression",
    verbosity=-1,
    num_threads=1,
    deterministic=True,
    force_col_wise=True,
    num_leaves=7,
    min_data_in_leaf=5,
    learning_rate=0.05,
    feature_pre_filter=False,
)
BOOST_ROUNDS = 24
