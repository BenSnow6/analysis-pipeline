### 1 • Status check — do the timestamp‑sanity results pass the “Week‑1 gate”?

**Yes, with flying colours.**

| Sensor                    | Jitter / gaps                            | Verdict                             |
| ------------------------- | ---------------------------------------- | ----------------------------------- |
| sensor 3 / 4 / 5 (200 Hz) | *0 ms* mean jitter, no gaps              | ★ Primary time base                 |
| sensor\_wb (100 Hz)       | < 0.3 ms jitter, no gaps                 | Good — align by 2 : 1 down‑sampling |
| GPS (1 Hz)                | Rare ≤ 20 ms jitter, a few ≤ 100 ms gaps | Acceptable with interpolation       |
| sensor\_wnb (\~7.5 Hz)    | 25 % rate error, large jitter            | **Exclude** for now                 |

The HTML/CSV/PNG reports already document these findings and can be pasted straight into the thesis Methods appendix.&#x20;

---

### 2 • What to do **next** (still inside Week 1)

Below is a **48‑hour task list** that completes the remaining Week‑1 deliverables without scope‑creep.

#### 2.1 Implement `align.py` — *Phase 1 (good sensors only)*

| Day        | Concrete tasks                                                                                                                                      | Time‑box |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **D1 AM**  | ▷ Create `align.py` with the `DataAligner` class stub from the Alignment Guide.  <br> ▷ Hard‑code `reference_sensor='sensor_3'`, `target_rate=200`. | 1 h      |
| **D1 AM**  | ▷ Implement `_align_sensor()` for **200 Hz** sensors using nearest‑neighbour match (≤ 2.5 ms tolerance).                                            | 1 h      |
| **D1 PM**  | ▷ Add branch for **100 Hz** (`sensor_wb`) — take every 2nd reference stamp.                                                                         | 1 h      |
| **D1 PM**  | ▷ Unit‑test with one real experiment: assert `len(aligned['sensor_wb']) == ½ len(ref)`.                                                             | 30 min   |
| **D1 PM**  | ▷ CLI wrapper: `python align.py -e <exp> -o aligned.hdf5`.                                                                                          | 30 min   |
| **D1 EOD** | **Commit & tag** `align_v0.1`.                                                                                                                      |          |

*Defer GPS and gap‑repair logic to Phase 2 next week to stay on schedule.*

#### 2.2 Validate sensor‑orientation matrix

| Day        | Concrete tasks                                                                                                                                                                                                       | Time‑box |              |   |   |   |                            |        |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ | - | - | - | -------------------------- | ------ |
| **D2 AM**  | ▷ Write `orientation_check.py`: <br>  • Load one static segment (first 5 s) of each IMU. <br>  • Transform raw accel with `sensor_orientations.json`. <br>  • Check that resulting **Z‑axis ≈ −9.81 m/s²**, X/Y ≈ 0. | 1 h      |              |   |   |   |                            |        |
| **D2 AM**  | ▷ Fail‑fast: any sensor with                                                                                                                                                                                         | Z        |  < 8 m/s² or | X | / | Y |  > 2 m/s² → print warning. | 30 min |
| **D2 PM**  | ▷ Quick Matplotlib scatter (no Seaborn) of gravity vector before/after transform for visual sanity.                                                                                                                  | 30 min   |              |   |   |   |                            |        |
| **D2 PM**  | ▷ Update `limitations.md` with any mis‑mounted sensor and planned workaround.                                                                                                                                        | 15 min   |              |   |   |   |                            |        |
| **D2 PM**  | ▷ Add **PyTest** case that feeds synthetic 9.81‑on‑Z data through every orientation matrix and expects corrected output.                                                                                             | 45 min   |              |   |   |   |                            |        |
| **D2 EOD** | **Commit & tag** `orientation_v0.1`; close Week‑1 Kanban cards.                                                                                                                                                      |          |              |   |   |   |                            |        |

---

### 3 • Quality guards to keep momentum

1. **Freeze scope:** `align.py` currently touches only 4 sensors. GPS & gap‑handling are Week 2 items—log them but don’t start.
2. **Unit tests first:** tiny tests for both `align.py` and orientation checker will stop silent regressions later.
3. **Runtime budget:** alignment of a 5‑min 200 Hz run should execute in < 1 s. Add an assertion.

---

### 4 • Where these tasks fit in the master plan

| Week‑1 deliverable                       | Status after these tasks                                |
| ---------------------------------------- | ------------------------------------------------------- |
| Clean, aligned data for **3 manoeuvres** | **Ready** for high‑rate IMUs; GPS pending               |
| Sensor‑orientation validation done       | **Ready**                                               |
| 10‑page Methods draft started            | Copy‑paste tool docs & orientation checks; expand later |

Complete these two steps, tag the commits, and you’re perfectly positioned to move into Week 2’s complementary filter work on Monday.