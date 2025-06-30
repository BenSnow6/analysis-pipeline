# Orientation Analysis Status & Next Steps

**Date**: 2025-06-18  
**Status**: Complete system ready with morning/afternoon separation

## 🎯 Quick Summary

We've built a complete analysis pipeline that:
1. ✅ Handles morning/afternoon sessions separately
2. ✅ Processes all experiments in `data/raw`
3. ✅ Uses truly static experiments for orientation validation
4. ✅ Prevents mixing of morning/afternoon data

## 📍 Where We Are

### ✅ What's Complete

1. **Full orientation analysis implementation** in `hovercraft_data_analysis/orientation_analysis/`:
   - Rotation matrix validation (without assuming correctness!)
   - Static segment detection with low-motion fallback
   - Dynamic maneuver validation
   - Sensor bias estimation
   - Cross-sensor consistency checks
   - Comprehensive visualization and reporting

2. **Morning/Afternoon data handling**:
   - Created `experiment_manifest.yaml` listing all experiments
   - Built `process_all_experiments.py` for batch processing
   - Created `run_static_orientation_analysis.py` for static experiments
   - Updated all scripts to handle both data structures

3. **Fixed all issues**:
   - ✅ Length mismatch between sensors (26001 vs 26000 rows)
   - ✅ Added gyro/angle/mag data alignment
   - ✅ Handle both IMU subfolder and direct sensor folder structures
   - ✅ Low-motion fallback when no static segments found

## 🚀 What to Do Tomorrow

### Step 1: Process All Evaluation Experiments (10-15 minutes)

```bash
cd C:\Users\ben\Documents\EngD\09 Data collection\01_analysis_pipeline\analysis-pipeline\hovercraft_data_analysis
python process_all_experiments.py
```

This will:
- Process all 22 experiments from `data/raw`
- Keep morning (9 experiments) and afternoon (13 experiments) separate
- Run alignment and add gyro/angle/mag data automatically

### Step 2: Run Orientation Validation on Static Data (5 minutes)

```bash
python run_static_orientation_analysis.py
```

This will:
- Use the truly static experiments (waiting periods)
- Validate sensor orientations for morning and afternoon separately
- Generate bias estimates for each session

## 📊 Expected Outputs

### Aligned Data Structure:
```
aligned_data/
├── morning/
│   ├── 006_Departure_aligned.h5
│   ├── 006_Departure_csv/
│   └── ... (9 experiments)
├── afternoon/
│   ├── 007_Fast_stbd_turn_1_aligned.h5
│   ├── 007_Fast_stbd_turn_1_csv/
│   └── ... (13 experiments)
└── static/
    ├── morning/
    │   └── 002_Setup, 004_Setup_2
    └── afternoon/
        └── 002_Setup, 003_Waiting_for_departure
```

### Orientation Results:
- Separate validation for morning/afternoon
- Rotation matrices for each session
- Bias estimates for each session
- Pass/fail for each sensor

## 💡 Key Insights

### Morning vs Afternoon Sessions
- Sensors were physically removed and reinstalled between sessions
- Each session has its own sync point and potentially different:
  - Mounting orientations (slight differences)
  - Sensor biases
  - Calibration parameters
- **Never mix morning and afternoon data!**

### Static Experiments for Validation
The best static data comes from waiting periods in `all_expts`:
- Morning: `002_Setup`, `004_Setup_2`
- Afternoon: `002_Setup`, `003_Waiting_for_departure`, `010_Waiting_for_static_turns`

These contain true static periods where the craft was sitting still.

## 📝 If Something Goes Wrong

1. **"Experiment not found"** → Check experiment_manifest.yaml for correct paths
2. **"No static segments found"** → Normal for dynamic experiments, uses low-motion fallback
3. **Length mismatch errors** → Already fixed by using sensor-specific timestamps
4. **Missing dependencies** → Run `pip install -r requirements.txt` in appropriate folders

## ✨ Ready for Tomorrow!

Just run the two main scripts:
1. `python process_all_experiments.py` - processes everything
2. `python run_static_orientation_analysis.py` - validates orientations

The entire pipeline is morning/afternoon aware and will keep your data properly separated!