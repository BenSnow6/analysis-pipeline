# Morning/Afternoon Data Processing Guide

## Important: Morning and Afternoon Sessions Must Be Kept Separate!

The hovercraft experiments were conducted in two sessions:
- **Morning**: Sensors installed, synchronized, waited for pilot, then experiments
- **Afternoon**: Sensors removed and reinstalled, new sync, waited for pilot, then experiments

**Critical**: Because sensors were reinstalled between sessions, they may have different:
- Physical orientations (slight mounting differences)
- Biases (sensor calibration can drift)
- Time synchronization points

## Directory Structure

```
02_Evaluation_Experiments/
├── 1a_1_Minimum_Radius_Turn/
│   ├── morning/
│   │   └── 015_Skirt_shift_turns/
│   └── afternoon/
│       ├── 007_Fast_stbd_turn_1/
│       ├── 009_Fast_port_turn_1/
│       └── ...
└── ...

all_expts/
├── morning/
│   └── Experiments/
│       ├── 002_Setup/          # Static - waiting with sensors installed
│       ├── 004_Setup_2/        # Static - more waiting
│       └── 006_Departure/      # Dynamic
└── afternoon/
    └── Experiments/
        ├── 002_Setup/                    # Static - waiting
        ├── 003_Waiting_for_departure/    # Static - waiting
        └── 010_Waiting_for_static_turns/ # Static - waiting
```

## Processing Pipeline

### 1. Process All Evaluation Experiments

```bash
cd hovercraft_data_analysis
python process_all_experiments.py
```

This will:
- Align all experiments in `02_Evaluation_Experiments`
- Keep morning and afternoon data separate
- Output to `aligned_data/morning/` and `aligned_data/afternoon/`

### 2. Run Orientation Validation on Static Data

```bash
python run_static_orientation_analysis.py
```

This will:
- Use truly static experiments from `all_expts` folder
- Validate sensor orientations separately for morning/afternoon
- Calculate bias estimates for each session

### 3. Apply Results

When processing data:
- Use morning rotation matrices and biases for morning experiments
- Use afternoon rotation matrices and biases for afternoon experiments
- Never mix data between sessions!

## Key Files

- `experiment_manifest.yaml`: Lists all experiments with session info
- `process_all_experiments.py`: Batch processes all evaluation experiments
- `run_static_orientation_analysis.py`: Runs orientation validation on static data

## Manual Processing

If you need to process specific experiments:

```bash
# Morning experiments
python alignment_analysis/run_alignment.py -e 015_Skirt_shift_turns -o aligned_data/morning

# Afternoon experiments  
python alignment_analysis/run_alignment.py -e 007_Fast_stbd_turn_1 016_Straight_cruise_1 -o aligned_data/afternoon

# Add gyro/angle/mag data
python alignment_analysis/align_additional_data.py

# Run orientation on morning static
python orientation_analysis/run_orientation.py -e 002_Setup 004_Setup_2 -d aligned_data/static/morning

# Run orientation on afternoon static
python orientation_analysis/run_orientation.py -e 002_Setup 003_Waiting_for_departure -d aligned_data/static/afternoon
```

## Verification

After processing, verify:
1. All experiments have been aligned
2. Morning and afternoon are in separate directories
3. Orientation validation found suitable rotation matrices
4. Bias estimates are reasonable (< 0.5 m/s² for accel, < 0.01 rad/s for gyro)

## Next Steps

Once alignment and orientation are complete:
1. Use validated rotation matrices for Kalman filtering
2. Apply session-specific bias corrections
3. Process each experiment with its session's calibration
4. Analyze results keeping sessions separate