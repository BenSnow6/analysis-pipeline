Below is a **lock‑step execution plan**—built from all the feedback but rewritten for **ADHD‑friendly, day‑to‑day use**—to take you from today to a *submitted, defensible EngD thesis* in 16 weeks.
Everything is expressed as *concrete check‑boxes, time‑boxes, and standing rituals* so you always know **exactly** what to do next.

---

## 0. Foundations (set up once, **Day 0–2**)

| Task                                                                                                                                               | Outcome                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Create a single Git repo** `hover-EngD-2025` with three top folders: `/code`, `/thesis`, `/docs`.                                                | One home for *everything*; no hunting.      |
| **Install & freeze tooling** (Python env with `requirements.txt`, Unreal version, VS Code, Zotero).                                                | Eliminate “which version?” surprises.       |
| **Make a 16‑week Kanban board** (Trello/Notion/Jira) with three columns only: Backlog → Doing (max 2 cards) → Done.                                | Immediate visual focus; zero overload.      |
| **Block your calendar**: 09:00‑12:30 “TECH”, 13:30‑16:30 “WRITE/EDIT”, 16:30‑17:00 “ADMIN”. Weekends free except *optional* 3‑hr catch‑up Sun p.m. | Same rhythm daily—ADHD brains love routine. |
| **Draft the thesis skeleton** (`/thesis/main.tex` or Word with built‑in style): title page, abstract, 6 chapter stubs, refs file.                  | Removes blank‑page anxiety.                 |
| **Set “red‑line” fallback triggers** (see §6).                                                                                                     | Decision guard‑rails—no dithering.          |

---

## 1. Month‑by‑Month Roadmap (high level)

| Month (Weeks)                      | Non‑Negotiable Deliverables (must be in **/docs/deliverables.md** by final Friday)                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M1 W1‑4** – “Ground Truth Ready” | ✅ Clean, time‑aligned CSVs for **3 manoeuvres**<br>✅ Complementary‑filter pitch/roll, GPS‑COG heading ready<br>✅ 10‑page Methods draft covering data pipeline |
| **M2 W5‑8** – “Sim ⇄ Real v1”      | ✅ Unreal reads RPM + steering CSV<br>✅ 3 overlay plots (speed, heading, trajectory)<br>✅ Parameter‑tuning notebook with auto‑RMSE output                      |
| **M3 W9‑12** – “Broader & Write”   | ✅ Same comparison for **1 extra manoeuvre**<br>✅ Error table vs. self‑defined tolerances<br>✅ Full Results & Discussion chapters draft                        |
| **M4 W13‑16** – “Polish & Submit”  | ✅ Code freeze tag `v1.0`<br>✅ Full thesis to supervisor (W13)<br>✅ Final PDF with all university formatting (W16)                                             |

Keep this table printed over your desk.

---

## 2. Weekly Sprint Template (every Monday 09:00 sharp)

1. **Pick exactly 2 cards** from Backlog → Doing.
   *One technical, one writing.*
   Add a **definition of done** line to each card.

2. **Write a “Friday Demo note”** (1 sentence × card) in `sprint_log.md`:
   “By Fri I will show …”.

3. **Daily stand‑up (self‑talk, 5 min, 09:00)**

   * What did I finish yesterday?
   * What blocks me?
   * Does anything break the 2‑card rule?
     Move/close cards immediately.

4. **Friday demo (16:00)**
   Paste plots, code diff, or chapter section into `sprint_log.md`.
   Move card to **Done**, tag commit, email supervisor two‑line update.

---

## 3. Daily Routine (times adjustable ±30 min)

| Time        | Activity                                                                     | Tools / Tips                                            |
| ----------- | ---------------------------------------------------------------------------- | ------------------------------------------------------- |
| 08:30       | **“Open Loop Dump”** – 10 min handwritten brain‑download of worries & ideas. | Clears mental RAM.                                      |
| 09:00‑12:30 | **TECH BLOCK** – strict **Pomodoro 50/10** ×3.                               | Noise‑cancelling headphones; put phone in another room. |
| 12:30‑13:30 | Lunch + 20 min **sunlight walk** (vitamin D, reset).                         | Physical movement combats ADHD slump.                   |
| 13:30‑16:30 | **WRITE/EDIT BLOCK** – pick next thesis paragraph or figure; 50/10 rhythm.   | Use **focus mode** (no code).                           |
| 16:30‑17:00 | **ADMIN/EMAIL & Kanban tidy‑up**.                                            | Prevent inbox creep into productive hours.              |
| Evening     | OFF. Exercise or social time.                                                | Rest ≠ laziness.                                        |

---

## 4. Detailed Task Lists (the *what* and *how*)

### Month 1 – Ground‑Truth Mastery

| Week   | Check‑List (tick as you go)                                                                                                                                                                                                 |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W1** | ✅ Run `quick_sanity.ipynb`: plot raw timestamps for one run; verify <20 ms jitter.<br>✅ Write `align.py` with **unit test** detecting gaps >100 ms.<br>~ Validate sensor orientations: gravity vector \~9.8 m/s² on Z‑body. |
| **W2** | ☐ Implement `comp_filter.py` → pitch/roll.<br>☐ Create `heading_proxy.py` (gyro + GPS COG).<br>☐ Produce **3 diagnostic plots**; paste PNGs into Methods draft.                                                             |
| **W3** | ✅ `rpm_fft.py` – Welch PSD on engine‑IMU; overlay known idle RPM.<br>☐ Decision gate: if SNR < 10 dB, plan fallback in `limitations.md`.<br>☐ 2‑hr spike on steering‑wheel IMU; if drift > 5 °/10 s, pivot.                 |
| **W4** | ☐ Assemble `input_files/*.csv` (time, RPM, steering/effective‑rudder).<br>☐ Lock **manoeuvre shortlist** in `experiments.json`.<br>☐ Submit 10‑page Methods draft to supervisor.                                            |

### Month 2 – Simulator Integration & First Comparison

| Week   | Check‑List                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------ |
| **W5** | ☐ Build minimal UE **CSV reader** (no UI).<br>☐ Run sim headless for 1 manoeuvre, export JSON states.                    |
| **W6** | ☐ Write `compare.py` → RMSE speed, heading, position.<br>☐ Generate first overlay plots; store in `/plots/v1`.           |
| **W7** | ☐ Create `tune_params.yml`; loop thrust, drag, rudder coeffs one‑at‑a‑time.<br>☐ Auto‑log best RMSE to `tuning_log.csv`. |
| **W8** | ☐ Freeze tuned params as `model_v2.yml`.<br>☐ Draft Results section for 3 manoeuvres (text + figures).                   |

### Month 3 – Extra Manoeuvre, Discussion, Intro/Conclusion

| Week    | Check‑List                                                                                                        |
| ------- | ----------------------------------------------------------------------------------------------------------------- |
| **W9**  | ☐ Run sim on **4th manoeuvre** without retune; compute errors.                                                    |
| **W10** | ☐ If any metric >2× tolerance, *one* focused retune cycle.<br>☐ Finalize acceptance table (`results_table.tex`).  |
| **W11** | ☐ Write Discussion: limitations, side‑slip caveat, steering fallback.<br>☐ Compile list of future‑work bullets.   |
| **W12** | ☐ Draft Introduction (problem, gap, contribution).<br>☐ Draft Conclusion (answers to research questions, impact). |

### Month 4 – Polish & Submission

| Week    | Check‑List                                                                                                                       |
| ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **W13** | ☐ Merge all chapters; run spell‑check & LaTeX/Word compile.<br>☐ Send full draft to supervisor.<br>☐ Code **freeze tag `v1.0`**. |
| **W14** | ☐ Address supervisor structural comments.<br>☐ Create `submission_checklist.md` (margins, figure numbering, ethics statement).   |
| **W15** | ☐ Final proof‑read aloud; fix typos.<br>☐ Generate archive `supplementary.zip` (data + code).                                    |
| **W16** | ☐ University online upload + hard‑copy (if required).<br>☐ Celebrate with 24‑hr dopamine‑rich activity of choice.                |

---

## 5. Living “Limitations & Decision Log”

Keep `/docs/limitations.md` open every day. For every pivot or simplification, add:

```
### <YYYY‑MM‑DD>  Steering IMU Abandoned
* Spent 6 hrs; drift 8 °/10 s – unacceptable.
* Adopted effective‑rudder tuning instead.
* Impact: cannot claim true input replay; handled in Discussion §4.3.
```

This becomes golden material for your viva.

---

## 6. Pre‑Declared **Fallback Triggers** (no agonising)

| Area                   | Trigger                         | Immediate Action                                            |
| ---------------------- | ------------------------------- | ----------------------------------------------------------- |
| **RPM inference**      | After 2 working days SNR <10 dB | Use **table‑lookup RPM** from manoeuvre notes; document.    |
| **Steering inference** | Gyro drift >5 °/10 s            | Switch to “match yaw‑rate” rudder tuning.                   |
| **Heading accuracy**   | GPS vs. gyro diverge >15 °      | Compare on **yaw‑rate** metric; note heading unreliability. |
| **Physics tuning**     | >3 days with RMSE unchanged     | Freeze current model; record as limitation.                 |
| **Writing lag**        | <3 pages produced in any week   | Drop lowest‑priority tech task the following week.          |

---

## 7. Personal ADHD Safeguards

1. **Body‑double sessions**: twice a week co‑work on Zoom with a friend; camera on, silent.
2. **Environment rotation**: every Wednesday TECH block in a library/café to refresh focus.
3. **Gamified streaks**: mark calendar with ✅ each day both TECH and WRITE blocks completed; 5‑day streak → micro‑reward (£10 treat).
4. **End‑of‑day shutdown ritual**:

   * Commit & push,
   * Close VS Code,
   * Write tomorrow’s first task on a sticky note,
   * Physically leave desk.

---

## 8. Communication Cadence

| Stakeholder                  | When                  | What                                |
| ---------------------------- | --------------------- | ----------------------------------- |
| Supervisor                   | Every Fri 16:05 email | 1 screenshot, 1 metric, 1 question. |
| External examiner (informal) | End of Month 2 & 3    | Short progress PDF.                 |
| Peer reviewer friend         | Month 3 Week 11       | Read Discussion draft for clarity.  |

---

## 9. Victory Conditions

* **Simulator** reproduces speed & heading within self‑set tolerances on 4 manoeuvres.
* **Thesis** 40 k–60 k words, all chapters complete, references compiled.
* **Submission** before Week 16 Friday 17:00.
* **You** still have mental and physical health intact.

Print these, tick them off, and *own your EngD journey*. You’ve got this.