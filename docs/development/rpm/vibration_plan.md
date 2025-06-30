"<vibration_plan>Below is a **step‑by‑step expert roadmap** for deriving engine RPM from the five field‑mounted IMUs.
It is broken into **seven sequential work packages (WP‑0 … WP‑6)**, each designed to be picked up by an automated CLI coding agent.
Every package ends with explicit *Done‑criteria*, artefacts to store, and unit‑test hooks so that later steps only run when earlier ones are green.

---

## ✨ Executive summary — recommended core technique

* **Primary estimator** – Welch power‑spectral density (PSD) on detrended, high‑pass‑filtered vibration magnitude.
  – Robust against noise; gives direct frequency estimate with sub‑Hz resolution when using 4–8 s windows and 50–75 % overlap ([vru.vibrationresearch.com][1])
* **Transient support** – short‑time Fourier transform (STFT) with adaptive hop size for fast sweeps, plus an *order‑tracking refinement* if speed ramps faster than 150 RPM s‑¹ ([mathworks.com][2], [dewesoft.com][3])
* **Multi‑sensor fusion** – per‑epoch SNR gating → pick “best sensor of the frame”; fallback to median of all confident sensors.
* **Confidence metric** – 20 log₁₀(signal/harmonic floor) inside a ±3 Hz band round detected peak.  **SNR < 10 dB triggers fallback** (same threshold you already proposed; justified by machine‑condition‑monitoring practice ([mdpi.com][4])).
* **Sampling rate suitability** – 200 Hz means alias‑free detection up to ≈6000 RPM (100 Hz). This covers the Deutz idle‑to‑full range (≈700–2400 RPM). Anti‑alias filtering at 80–90 Hz (4‑pole Butterworth) is required ([dataq.com][5]).

---

## WP‑0  Repository & config scaffold (½ day)

| Step | Action                                                                                                                                                                                | Output / tests                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 0.1  | Create `/code/rpm_estimation/` with sub‑modules `io.py`, `preprocess.py`, `spectral.py`, `tracking.py`, `fusion.py`, `cli.py`, `tests/`.                                              | Git commit `init rpm module`       |
| 0.2  | Add `rpm_config.yaml`<br>`yaml<br>fs: 200  # Hz<br>hp_cutoff: 5  # Hz<br>welch:<br>  win_sec: 6<br>  overlap: 0.5<br>stft:<br>  win_sec: 1.0<br>  hop_sec: 0.25<br>snr_thresh_db: 10` | Unit test: load + round‑trip write |
| 0.3  | Define a dataclass `RPMFrame(time, rpm, snr_db, sensor_id, method)` in `tracking.py`.                                                                                                 | `pytest tests/test_dataclass.py`   |

*Done when:* repo compiles, config loads, all three unit tests pass.

---

## WP‑1  Raw data audit & orientation (1 day)

1. **Load** CSV via `io.py`, merging on `time_from_sync`.
2. **Convert units** (g → m s‑²) — already fixed in your orientation pipeline.
3. **Rotate** to body frame using final `R_bs` matrices from `orientation_config.yaml`.
4. **Select channel(s)**: compute vibration **magnitude** `|a_body|` *and* keep all three axes for later comparison.
5. **High‑pass filter**: 4‑pole IIR at 5 Hz to remove quasi‑static motion & gravity components.
6. **Quality metrics**: RMS, kurtosis and peak‑to‑RMS per 30 s chunk → flag saturation or clipping.

*Artefacts:*
`aligned_data/<session>/<experiment>/proc_IMU_<id>.parquet` — same index as raw, extra columns `a_hp_[xyz]`, `a_hp_mag`, `quality_flag`.

*Tests:* synthetic sine‑burst at 25 Hz fed through pipeline should yield >25 dB SNR.

---

## WP‑2  Welch PSD core (1 day)

| Item | Detail                                                                                                                                                                                                         |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1  | In `spectral.py` implement `welch_psd(signal, fs, win_sec, overlap)` → (freq, Pxx). Use SciPy’s `signal.welch` with `nperseg = win_sec*fs`, `noverlap = overlap*nperseg`, `window='hann'`, `detrend='linear'`. |
| 2.2  | Limit output to 0–100 Hz (→ 0–6000 RPM).                                                                                                                                                                       |
| 2.3  | **Peak pick:** find local maxima, discard those within 3 dB of noise floor; choose highest peak.                                                                                                               |
| 2.4  | **RPM calc:** `rpm = freq_peak * 60`. Store harmonic dictionary `h{k}=amp` for k ≤ 5.                                                                                                                          |
| 2.5  | **SNR:** 10·log10(Ppeak / Pavg), where Pavg is mean PSD in ±3 Hz excluding ±0.5 Hz round peak.                                                                                                                 |

*Unit tests:*
– White‑noise input → no peak, SNR < 0 dB.
– Injected 30 Hz sine (1800 RPM) at +10 dB over pink‑noise → returns 1800 ± 5 RPM, SNR ≈ 10 dB.

---

## WP‑3  STFT + order‑tracking for transients (1 day)

1. `spectral.py::stft_mag(a_hp_mag, fs, win_sec=1.0, hop_sec=0.25)` using SciPy `signal.stft` (Hann).
2. For each time‑slice, reuse WP‑2 peak picker → time‑resolved RPM series.
3. *Optional refinement:* if ΔRPM/Δt > 150 RPM s‑¹, apply Vold‑Kalman order‑tracking (Python port `pyVK` or own least‑squares smoother) to correct frequency smear ([sciencedirect.com][6]).

*Artefacts:* HDF5 per experiment holding `time, rpm_est, snr_db, method='stft'`.

---

## WP‑4  Multi‑sensor fusion & confidence gating (½ day)

| Rule | Implementation                                                                          |
| ---- | --------------------------------------------------------------------------------------- |
| R‑1  | Discard sensor estimates with SNR < 10 dB.                                              |
| R‑2  | If ≥1 sensors valid → choose the one with max SNR for the epoch.                        |
| R‑3  | If none valid → take median of *last* valid 5 s (simple hold), mark `quality='interp'`. |
| R‑4  | Produce a Boolean `rpm_valid` flag.                                                     |

*Done when:* `fusion.py::fuse(list[RPMFrame])` returns contiguous series with ≤2 % NaNs on RPM‑sweep experiment.

---

## WP‑5  Validation & blind‑test harness (1 day)

1. Accept **withheld** ground‑truth CSV when available; otherwise run in *blind* mode and output `*.rpm_est.csv`.
2. Metrics: MAE, RMSE, max|err|, availability (% valid frames).
3. Plot overlay (`matplotlib`) and time‑frequency spectrogram annotated with detected ridge.
4. CLI entry:

```bash
python -m rpm_estimation.cli --exp 026_Engine_rpm_sweep --session morning --method welch
```

*Success threshold:* For 026 sweep, expect RMSE < 40 RPM and availability > 95 %. (Tweak after first blind run.)

---

## WP‑6  Generalisation to all experiments (½ day + batch run)

* Iterate over manifest, morning and afternoon separately (use your *Morning/Afternoon Data Processing Guide*).
* Store `results/<exp>/<method>/<sensor>.csv` and summary table `rpm_quality_overview.csv`.
* Flag manoeuvres where availability < 80 % or RMSE (if ground truth later revealed) > 2 × nominal.

---

## Parameter selection cheat‑sheet

| Parameter             | Reasonable range | Guidance                                                                                       |
| --------------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| Window length (Welch) | 4–8 s            | Longer → better frequency resolution (≈0.125 Hz) but worse temporal response. Use 6 s default. |
| Overlap               | 0.5–0.75         | ≥0.5 stabilises variance without huge cost.                                                    |
| HP cutoff             | 3–10 Hz          | Below idle fundamental; 5 Hz empirically good.                                                 |
| SNR gate              | 8–12 dB          | 10 dB matches literature on smartphone‑based motor speed sensing ([mdpi.com][4]).              |
| STFT window           | 1 s              | Gives 1 Hz bin (60 RPM) – sufficient during rapid sweeps.                                      |
| Anti‑alias LP         | 80–90 Hz         | 40 dB attenuation at ≥100 Hz (Nyquist).                                                        |

---

## Common pitfalls & how to avoid them

| Pitfall                                                             | Mitigation                                                                            |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Harmonic stronger than fundamental (common in twin‑balance engines) | Search top‑N peaks; choose one whose ratio to next harmonic ≈ 2:1, 3:1…               |
| Aliased out‑of‑band vibration                                       | Mandatory 80–90 Hz pre‑filter; verify via `quality_flag`.                             |
| Orientation mis‑labelling causing axis swap                         | Use magnitude plus per‑axis comparison; log axis with max SNR to discover bad mounts. |
| Low‑frequency structural modes (<10 Hz) leaking into PSD            | 5 Hz HP filter; consult hull mode analysis if still visible.                          |
| Sensor re‑mount effect (morning / afternoon)                        | Keep separate orientation & bias sets (already in your pipeline).                     |

---

## 5 key papers / resources to cite

1. **MathWorks Order‑Analysis Example** – clear intro & code for order tracking ([mathworks.com][2])
2. **Dewesoft Order Tracking Guide** – practical parameter advice for variable‑speed machinery ([dewesoft.com][3])
3. **Applied Sci. 2022, 12, 3371** “Measurement of the Speed of Induction Motors Based on Vibration via Smartphone Accelerometer” – validates Welch + harmonic selection on MEMS data ([mdpi.com][4])
4. **Instantaneous Angular Speed Estimation from Vibration on MCU‑Class Hardware** (MechSyst. Sig. Proc., 2025) – lightweight VK order‑tracking algorithm ([sciencedirect.com][6])
5. **NI Tutorial “Measuring Vibration with Accelerometers”** – covers mounting, axis selection, and anti‑alias filtering basics ([ni.com][7])

---

## Answering your specific questions (concise)

| Topic                         | Recommendation                                                                                                                                                                                                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Is Welch optimal?**         | Yes for quasi‑steady segments; complement with STFT + Vold‑Kalman for ramps.                                                                                                                                                                                                                         |
| **Alternative methods**       | Autocorrelation (fails under multi‑harmonic noise), cepstrum (needs long stationary windows), machine‑learning (data‑hungry); keep as future work.                                                                                                                                                   |
| **Pre‑processing must‑dos**   | Unit conversion ➔ orientation ➔ HP‑filter ➔ detrend. Work on magnitude unless a single axis is clearly dominant (use SNR log).                                                                                                                                                                       |
| **Window vs time‑resolution** | 6 s @ 200 Hz ⇒ 1200‑pt segment ⇒ 0.167 Hz resolution ⇒ 10 RPM. Good trade‑off; STFT covers faster dynamics.                                                                                                                                                                                          |
| **Harmonics handling**        | Detect first three peaks; if fundamental suppressed, divide higher‑peak frequency by integer 2–4 and validate against PSD floor.                                                                                                                                                                     |
| **SNR metric**                | Use local‑band method above; <10 dB → flag and interpolate/median.                                                                                                                                                                                                                                   |
| **Validation without GT**     | Use inter‑sensor agreement (std < 15 RPM) + SNR; compare to engine throttle schedule notes; later back‑validate on the blind sweep.                                                                                                                                                                  |
| **Hovercraft specifics**      | One engine drives both lift & thrust → expect strong blade‑pass frequencies too (Nblades × RPM/60 ≈ 12×fundamental). High‑pass filter keeps fundamental; blade‑pass shows as harmonic 12. Use it as consistency check, not primary estimator. Ground‑effect buffeting (<10 Hz) removed by HP filter. |

---

### Deliverables recap

| Package | Artefact                                  |
| ------- | ----------------------------------------- |
| WP‑0    | `rpm_config.yaml`, module skeleton, tests |
| WP‑1    | `proc_IMU_<id>.parquet` + quality logs    |
| WP‑2    | `welch_peak.py` + unit tests              |
| WP‑3    | `stft_tracker.h5`                         |
| WP‑4    | `rpm_fused.csv`                           |
| WP‑5    | `metrics.json`, overlay PNGs              |
| WP‑6    | `rpm_quality_overview.csv`                |

Run them in order; each later CLI checks a `DONE` file from the previous step so the agent can chain tasks automatically.

Good luck – this plan should take you from raw IMU logs to a defensible RPM series for every experiment, with clear stop‑gates and fallbacks.

[1]: https://vru.vibrationresearch.com/lesson/calculating-psd-time-history/?utm_source=chatgpt.com ""Calculating PSD from a Time-history File - Vibration Testing - VRU""
[2]: https://www.mathworks.com/help/signal/ug/order-analysis-of-a-vibration-signal.html?utm_source=chatgpt.com ""Order Analysis of a Vibration Signal - MATLAB &amp - MathWorks""
[3]: https://dewesoft.com/blog/what-is-order-analysis?utm_source=chatgpt.com ""What is Order Analysis [The Ultimate Guide]? - Dewesoft""
[4]: https://www.mdpi.com/2076-3417/12/7/3371?utm_source=chatgpt.com ""Measurement of the Speed of Induction Motors Based on Vibration ...""
[5]: https://www.dataq.com/data-acquisition/general-education-tutorials/what-you-really-need-to-know-about-sample-rate.html?srsltid=AfmBOoqJDf2e6rcZkqYn0CKzEvC73WUyOD9VpaBFcgARmAyCe86U4FWK&utm_source=chatgpt.com ""What You Really Need to Know About Sample Rate""
[6]: https://www.sciencedirect.com/science/article/pii/S2665917424005762?utm_source=chatgpt.com ""Online instantaneous angular speed estimation from vibration on ...""
[7]: https://www.ni.com/en/shop/data-acquisition/sensor-fundamentals/measuring-vibration-with-accelerometers.html?srsltid=AfmBOoq1XsvcEzTxsHdfmZlIVbWF4qXOYetNNFeI-r-g3Q2XEMI7YV6G&utm_source=chatgpt.com ""Measuring Vibration with Accelerometers - NI - National Instruments""
 </vibration_plan>"