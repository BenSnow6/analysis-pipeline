# Combined & Refined Plan for Chapter 3: Methodology: Implementation of a Real-Time Hovercraft Simulator

**(Target Word Count: ~10,000 - 13,000 words –  Core sections (3.3-3.5) ≈ 55-60%.)**

**Overall Goal:** To provide a clear, detailed, and replicable account of the development and implementation of the hovercraft simulator within Unreal Engine, justifying design choices, specifying parameters and their provenance, detailing algorithms and models, and structuring the description according to an adapted ICAO FSTD framework, explicitly addressing the requirements for an EngD thesis regarding engineering rigour and potential for future extension.

---

## Chapter Structure & Content Plan:

### 3.1 Introduction
*   **Purpose:** Briefly restate the overall thesis aim (developing a validated, real-time hovercraft simulator). State the specific purpose of *this* chapter: to detail the *how* of the simulator's construction, bridging the theoretical foundations (Chapter 2: Why) and the empirical testing (Chapter 4: How Well).
*   **Framework & Adaptation:** Explicitly state the use of the adapted ICAO Document 9625 framework. **Crucially, adapt terminology** for the hovercraft context (e.g., "Flight Model" becomes "Vehicle Dynamics Model"). Justify this choice (provides a comprehensive, recognized structure for simulator description).
*   **Implementation Compliance Table (NEW - Per Plan 2):** Insert Table 3.1 early, listing the 14 ICAO features with columns: Feature | Adapted Hovercraft Term | Implementation Level (✔ Implemented / △ Scaled-down / ✖ Omitted) | Brief Rationale for Level/Omission (linked to training scope/fidelity target, e.g., DNV-C functional requirements). *This proactively addresses scope questions.*
*   **Development Platform & Tools:** Reiterate Unreal Engine choice (justified in Ch2). **Specify UE Version (e.g., 5.X.Y), crucial plugins (e.g., Cesium for Unreal [Version], MMT Water [Version], relevant UE Marketplace assets), source control details (e.g., Git repository snapshot tag/commit hash).** Mention primary development language(s) (Blueprints, C++). List key engine features leveraged (e.g., Chaos Physics, Blueprints, C++, UI Widgets).
*   **Architectural Overview:** Briefly mention key architectural decisions (e.g., component-based design within UE Actor hierarchy, use of physics sub-stepping for deterministic calculations - detailed later).
*   **Ethics/Licensing Note:** Briefly state compliance with licenses for used assets/data (e.g., Cesium/Google Maps photogrammetry terms of service, audio assets, 3D models – educational/commercial license as applicable).
*   **Roadmap:** Outline the sections (3.2-3.11), noting adaptations and merged sections.

### 3.2 Cockpit Layout and Structure
*   **ICAO Feature Adaptation:** Define purpose – virtual operator environment representation.
*   **Implementation:**
    *   **3D Model:** Describe the source/creation of the 3D cockpit model (e.g., adapted from existing asset, custom-modelled in Blender). **State target fidelity explicitly** (e.g., "Representative fidelity sufficient for DNV-C functional requirements, prioritizing control layout visibility and key visual cues over photo-realistic replication of the 2000TD").
    *   **Instrumentation (HUD):** Detail the implementation using UE Widget Blueprints. Specify variables displayed (RPM, speed, pitch/roll angles, compass heading, control positions, etc. – cross-ref Visual Display section). Mention source of reference photos (e.g., 2000TD manuals/images).
    *   **User Roles & Interface:** Describe the setup for Trainee and Instructor roles (even if sharing one interface). Detail Instructor capabilities accessible via UI menus (e.g., scenario initiation, environmental controls - ref IOS section).
    *   **Hardware Integration:** Describe the physical setup (single monitor, PC, chair). Mention the use of generic peripherals (steering wheel, pedals, joystick) and how their axes/buttons conceptually map to hovercraft controls (details in Flight Controls section).
*   **Validation Hook:** *"The accuracy and responsiveness of the HUD displaying key vehicle state parameters are verified against logged data in Section 4.X."*

### 3.3 Vehicle Dynamics Model (*Core Section 1 - High Detail*)
*   **ICAO Feature Adaptation:** Defining the mathematical models and algorithms governing the hovercraft's six-degree-of-freedom motion.
*   **Implementation Architecture:** Describe the structure within UE (e.g., custom C++ Actor Component attached to the main hovercraft Pawn/Actor). Clarify integration with UE physics tick and the role of physics sub-stepping (detailed in Misc section).
*   **Powertrain:**
    *   *Engine Model:* Detail the simplified engine RPM model: mapping input lever (0-1 range) to target RPM (0-2100 RPM); implementation of PD controller for smooth RPM changes; rate limiting application. State assumptions (e.g., single global RPM drives both lift and thrust proportionally, instantaneous response negligible).
    *   *Propulsion Model:* Describe propeller thrust calculation: reference PPC equations (cited from Ch2); implementation of Thrust = f(RPM, Pitch) relationship (e.g., Thrust97 curve fit). Detail how thrust vector magnitude and direction are modified by rudder and elevator deflections (vector math description). Describe reverse pitch implementation (e.g., activation logic, thrust magnitude reduction factor). *Include simplified flowchart/pseudocode for thrust calculation.*
*   **Lift Model (N-Tile Cushion Model):**
    *   *Conceptual Basis:* Reference Reynolds '72 justification (Ch 2).
    *   *Implementation Details:* Detail the N-tile spring/damper approach: specify number, layout, and distribution of pressure points/elements under the hull (include clear diagram). Describe interaction method: how line traces/sphere casts (specify UE function used) detect distance to ground/water surface beneath each element. Detail cushion pressure calculation: how pressure is calculated/interpolated based on element compression height, potentially modulated by engine RPM state (e.g., simple pressure-RPM map or link to fan performance curve if modelled). Describe spring/damping implementation: how forces are calculated based on element compression and velocity.
*   **Drag Model:**
    *   *Implementation Details:* Detail how *each* significant drag component identified in Ch 2 was implemented in code/Blueprints.
    *   *Aerodynamic Drag:* Describe real-time projected area calculation (e.g., referencing pre-calculated CSV data based on Azimuth/Elevation angles – detail the lookup method). How drag coefficient (Cd) is applied. Clarify distinction between wind drag (based on relative wind vector) and motion drag (based on velocity vector) implementation.
    *   *Wavemaking Drag:* Detail the implementation based on Barratt (or chosen method from Ch2). Explain how craft orientation and speed inputs are used. Describe implementation for runtime efficiency (e.g., lookup tables, UE float curve assets). Discuss pitch dependency explicitly (was it implemented? If not, justify omission – e.g., complexity vs. impact at target speeds).
    *   *Momentum Drag:* Describe the implementation logic (e.g., simplified model based on air inflow/outflow momentum change calculation, potentially linked to projected area or a fixed coefficient).
    *   *Skirt Contact Drag:* Describe implementation logic: conditions for activation (e.g., skirt element ground contact detected by N-tile system). Detail frictional drag calculation (coefficients used – sourced/tuned? Simplified linear/Coulomb friction model?). Mention potential use for low-speed control/damping.
    *   **Justification for Neglected Drags:** Explicitly state which minor drag components (e.g., spray drag) were neglected and provide justification (e.g., "Estimated contribution < X% of total drag under typical operating conditions based on [cite source/estimation], considered negligible for target fidelity").
*   **Parameter Provenance Table (NEW - Crucial):** Insert Table 3.X (or reference Appendix) listing key dynamics parameters: Symbol | Description | Value | Units | Source (Literature [Cite], Derived [Explain], Tuned [Ref Ch4/5 Validation], Assumed) | Uncertainty/Sensitivity Note (if applicable). *Examples: All-Up Weight, Moments of Inertia (how derived? CAD/simplified?), CoM base location, N-tile spring/damping constants, drag coefficients, PPC parameters. Flag tuned parameters explicitly.*
*   **Force Application:** Specify precisely where each calculated force vector (lift elements, thrust, drag components) is applied to the hovercraft's rigid body component in Unreal Engine (e.g., Center of Mass, specific offset points for thrust/lift elements).
*   **Stability Implementation:**
    *   *Pitch/Roll Stability:* Describe implementation of restoring moments (e.g., derived naturally from N-tile pressure distribution? simplified torque proportional to roll/pitch angle? combination?). Mention plough-in simulation limits (e.g., conditions where restoring moments are insufficient).
    *   *Heave Stability:* Explain how it emerges from the N-tile cushion model's effective stiffness and damping characteristics.
*   **Control Systems Effects (Implemented within Dynamics):**
    *   *Fuel Ballasting:* Describe the CoM shifting logic implementation (input mapping, max travel distance/rate, axis of movement). Justify simplifications (e.g., less frequently used, trim typically set initially).
    *   *Skirt Shifting:* Describe implementation method (e.g., modelled as Centre of Pressure shift? analogue CoM shift? differing N-tile stiffness based on button press?).
    *   *Payload Movement:* Describe logic implementation (similar CoM shift logic to fuel ballast but potentially larger mass/different movement axes).
*   **Numerical Verification & Robustness (NEW Subsection - Per Plan 2):**
    *   Describe simple internal verification tests performed during development (e.g., check for energy conservation/drift during static hover over 60s, basic step-size sensitivity analysis for physics sub-stepping, force balance checks).
    *   Briefly mention implemented safeguards against numerical instability (e.g., NaN checks on calculated forces, value clamping, safe default values for parameters).
*   **Validation Hooks:** Add pointers for key sub-models, e.g., *"The accuracy of the lift model's predicted hover height versus RPM is assessed against reference data in Section 4.X.Y."*, *"The fidelity of the wave drag model during acceleration and deceleration manoeuvres is validated in Section 4.A.B."*, *"Overall vehicle stability characteristics are evaluated through simulated manoeuvres in Chapter 4."*

### 3.4 Surf and Buoyant Handling (*Core Section 2 - Medium Detail*)
*   **ICAO Feature Adaptation:** Defining ground/water contact dynamics when off-cushion.
*   **Implementation:**
    *   **Buoyancy Model:** Describe the method used (e.g., multiple sphere test points simulating pontoons). Specify location, number, and radius of simulated spheres. **Justify choice over built-in UE Buoyancy** (e.g., "Provides explicit control over force application points and calculation logic, deemed more transparent for debugging and tuning than the integrated module for this specific application"). Detail how buoyancy force is calculated based on each sphere's submersion depth (Archimedes principle implementation). Include calculation flowchart/pseudocode if complex.
    *   **Water Interaction:** Describe how the simulator interacts with the UE Water system plugin (e.g., querying water height at pontoon/N-tile locations using specific UE functions). State key assumptions (e.g., craft does not deform water surface, water physics interaction is one-way height query).
    *   **Transition Logic:** Detail how the simulator detects and handles transitions between surf/flight states (e.g., based on cushion pressure thresholds, N-tile ground contact flags, minimum hover height).
    *   **Hydrodynamic Forces (Off-Cushion):** Detail any specific drag or damping forces applied only when pontoons are significantly submerged (if different from airborne drag model, e.g., simplified viscous drag based on pontoon submersion/velocity).
*   **Validation Hook:** *"Buoyant force calculation accuracy (e.g., static flotation attitude) and the dynamic behaviour during surf-to-flight transitions are evaluated in Section 4.X."*

### 3.5 Hovercraft Systems Simulation (*Core Section 3 - Concise*)
*   **ICAO Feature Adaptation:** Defining the scope and level of onboard systems simulation (hydraulics, electrical, fuel, etc.).
*   **Implementation:**
    *   **Scope Statement:** Explicitly state compliance level: *"Detailed simulation of onboard systems (e.g., hydraulics, electrical distribution, fuel flow, engine sub-components, failure modes) was deemed outside the scope of this project (Compliance Level: Omitted/Minimal, see Table 3.1). This is justified as the primary training objectives (Section [Ref Intro/Training Needs Analysis]) focus on fundamental vehicle handling skills and spatial awareness, not complex system management or emergency procedures."*
    *   **Input Filtering:** Detail the simple filtering applied to raw peripheral inputs before use by the dynamics model (e.g., implementation of PD controllers or rate limiters on control surface demands) to ensure smooth response – describe implementation method and parameters (e.g., gains, limits).
*   **Validation Hook:** *"While detailed systems are not modelled, the responsiveness and handling qualities resulting from the filtered control inputs are assessed during manoeuvre-based validation trials in Chapter 4."*

### 3.6 Flight Controls and Forces
*   **ICAO Feature Adaptation:** Mapping physical operator controls to simulation inputs, and simulation of control forces (feel).
*   **Implementation:**
    *   **Input Mapping:** Detail how physical controller inputs (axes, buttons from steering wheel, pedals, joystick etc.) are mapped to simulation control variables (engine RPM lever demand, propeller pitch demand, rudder angle demand, elevator deflection demand, skirt shift commands) using the Unreal Engine input system (e.g., Input Actions, Axis Mappings).
    *   **Sensitivity & Response:** Describe how input sensitivity curves, scaling factors, or dead zones were defined and tuned to achieve desired craft responsiveness. Note handling of event-based triggers (buttons) vs continuous inputs (axes).
    *   **Control Forces (Feel):** State clearly that realistic control *feel* (force feedback) was **not implemented**. **Justify** this omission by linking to the target fidelity level (e.g., functional simulation for DNV-C), the training scope (basic handling skills), the desktop hardware platform, and potentially citing literature suggesting visual/vestibular cues are dominant for primary control in this context.
*   **Validation Hook:** *"The effectiveness of the control mapping and sensitivity tuning in enabling precise vehicle control is evaluated qualitatively and quantitatively across multiple standardized manoeuvres in Chapter 4."*

### 3.7 Visual Display Cue
*   **ICAO Feature Adaptation:** Defining the simulated out-of-cockpit visual scene and supporting display elements.
*   **Implementation:**
    *   **Out-of-Cockpit View:** Describe the standard UE camera setup used (e.g., perspective camera attached to cockpit). Specify Field of View (FOV) settings and justification (e.g., balancing immersion and peripheral awareness).
    *   **HUD:** Reiterate implementation (UE Widget Blueprint). Cross-reference Cockpit section (3.2) for variables displayed. Discuss any specific design choices made for clarity or usability (e.g., layout, colour coding).
    *   **Environmental Effects (Visual):**
        *   *Wake:* Describe the logic for spawning wake visual effect actors or particle systems (e.g., conditions based on speed/propeller state, position relative to stern, lifetime, visual appearance).
        *   *Spray:* Describe the logic for activating spray particle systems (e.g., conditions based on trim angles, speed, cushion state). Specify particle system characteristics (e.g., material, emission rate, lifespan, velocity).
    *   **Cockpit Visuals:** Mention any specific lighting applied within the cockpit model (e.g., simple plane light for instrument visibility). Describe any window glass effects implemented (e.g., simple reflections, dirt/water effects if any).
*   **Validation Hook:** *"The adequacy of the visual cues for situational awareness and vehicle control is assessed primarily through pilot feedback and task performance metrics during validation trials (Chapters 4 & 5)."*

### 3.8 Non-implemented or Minimal-Fidelity Cues (Merged Section)
*   **ICAO Features Covered:** Sound, Vibration, Motion.
*   **Implementation Level & Justification:**
    *   *Sound:* Briefly describe the minimal implementation (e.g., basic engine audio loop with pitch and volume modulated by calculated engine RPM using UE audio components, ambient ocean wave sounds sourced from [Specify Source]). State scope limitation (e.g., "High-fidelity, spatially accurate audio simulation was not a requirement").
    *   *Vibration:* State "**Not Implemented**". Justify (desktop simulator platform, limited benefit for target training objectives/fidelity level).
    *   *Motion:* State "**Not Implemented**". Justify (desktop simulator platform).

### 3.9 Environment Simulation (Consolidated Grouping)
*   **ICAO Feature Adaptation:** Defining the simulated external world including navigation, weather, terrain, and ATC elements.
*   **Implementation:**
    *   **3.9.1 Navigation:** (ICAO: Environment — Navigation). Describe the use of a georeferenced pawn within the UE/Cesium environment for positioning. Detail how simulated GPS data (Latitude/Longitude/Altitude) is generated from the pawn's world position. Describe implementation of navigational displays (e.g., simple overhead map view UI widget, compass rose integrated into HUD).
    *   **3.9.2 Weather:** (ICAO: Environment — Weather).
        *   *Lighting:* Describe the day/night cycle implementation (e.g., driven by simulation time or manually set via IOS). Detail use of UE Sky Atmosphere / Ultra Dynamic Sky (or similar) system. Mention PBR material interaction (roughness affecting reflections, absorption). Note dynamic shadow effects (including from Cesium buildings).
        *   *Wind:* Detail the simple wind model implemented (e.g., uniform global wind vector applied as a force to the craft's aerodynamic model). Describe how wind direction/speed are controlled (e.g., via IOS menu).
        *   *Ocean State:* Explain how the Beaufort scale setting (selectable via IOS) translates to parameters controlling the UE Water system (e.g., wave height, speed, length, chop intensity, direction).
        *   *Visibility:* Describe use of volumetric fog or similar UE features to control visibility distance (controlled via IOS).
        *   *Visual Effects:* List key post-processing effects used to enhance atmospheric realism (e.g., Bloom, Lens Flare, Exposure Compensation, Color Grading - White Balance, Saturation).
    *   **3.9.3 Landing Areas & Terrain:** (ICAO: Environment — Landing Areas and Terrain).
        *   *Georeferenced World:* Detail the use of the Cesium for Unreal plugin for streaming real-world terrain and 3D building photogrammetry tiles (specify data source, e.g., Cesium World Terrain, Bing Maps).
        *   *Terrain Interaction:* Describe how the ocean plane interacts visually with the streamed terrain (e.g., wave clipping/shoreline). Note the dynamic loading/LOD system inherent in Cesium. **Mention tile caching strategy and deterministic LOD seeding efforts for reproducibility between sessions if applicable.**
        *   *Custom Areas:* Detail the modelling and integration of any specific custom areas, like the GHL Woolston slipway (modelling software, source data like satellite imagery, scaling process, integration into Cesium world).
        *   *Lighting on Terrain:* Mention use of baked lighting derived from real-world data for static elements like buildings (inherent in Cesium tiles).
    *   **3.9.4 ATC:** (ICAO: Environment — ATC). State clearly "**Not Implemented**". Justify (no requirement for air traffic control interaction in the hovercraft operational context and training scope).
*   **Validation Hooks:** *"The influence of selectable ocean states (Beaufort levels) on vehicle dynamics and controllability is tested in Section 4.X."* or *"The fidelity and usability of the visual environment are implicitly assessed through pilot task performance and subjective feedback during validation trials (Chapter 4/5)."*

### 3.10 Miscellaneous Supporting Features
*   **ICAO Feature Adaptation:** Covering supporting functionalities like the Instructor Operating Station (IOS), data logging, diagnostics, replay, etc.
*   **Implementation:**
    *   **Instructor Operating Station (IOS):** Describe the UI menus developed using UE Widgets. Detail functionalities provided: basic settings menu (e.g., dynamics tuning parameters like heave stiffness multipliers?), Environment control menu (Sea state/wind/time of day/visibility), Simple scenario controls (start/stop/reset), potentially a basic tutorial/controls reminder display.
    *   **Data Collection:** Detail the system implemented for logging key craft state variables and performance metrics: parameters logged (position, orientation, velocities, accelerations, control inputs, RPM, cushion state, etc.), logging frequency (e.g., every physics tick? fixed rate?), data format (e.g., CSV), file naming convention, storage location.
    *   **Replay System:** Describe the implementation approach (e.g., recording input controls and initial state? recording pawn transform and key state variables over time?). Detail playback capabilities (e.g., ability to re-watch simulation run, potentially from different camera views like external observer).
    *   **Feedback System (Post-Simulation Analysis):** Detail the process for generating feedback reports from logged CSV data: describe the tool/method used (e.g., external Python script using libraries like Pandas/Matplotlib, integrated analysis within UE?). Specify key metrics calculated and plotted in reports (e.g., pitch/roll time histories, yaw rate analysis, track plots, highlighting excursions outside safe operating envelopes, comfort metric calculations, time spent in hump transition zone, trim estimation).
    *   **Physics Sub-stepping:** Explain *how* physics sub-stepping was configured and implemented in Unreal Engine's project settings and/or custom code. Clarify the motivation (achieving deterministic physics calculations independent of rendering frame rate). Mention the physics thread setup. **Include performance comparison table/notes showing achieved simulation frequency (Δt) and rendering FPS on Development Rig vs. Target Low-Specification Machine (e.g., 'old laptop'), noting any observed behavioural differences.**
    *   **Development & Deployment:** Briefly mention PC specifications used for primary development. Note testing conducted on lower-spec hardware. Describe input device management approach (e.g., designed for input agnosticism via UE input mapping). Outline the patching/distribution strategy used during development and testing (e.g., packaging builds into .pak files).
    *   **Reproducibility Note:** Explicitly state code and potentially key asset availability (e.g., "Core C++ algorithms and Blueprint logic available at [Link to Public GitHub Repository / Institutional Archive]" or "Code archived with University Research Data Repository [See Appendix X for Identifier/Access Instructions]").

### 3.11 Chapter Summary
*   Briefly recap the key methodological choices made during the simulator development (e.g., adapted ICAO structure, UE platform, N-tile model, Cesium integration, sub-stepping).
*   Succinctly reiterate the rationale for the achieved fidelity level and the scope defined by the project requirements and training objectives.
*   Provide a clear transition sentence stating that the *following chapter (Chapter 4)* will detail the experimental design, procedures, and results used to validate the performance, fidelity, and utility of the hovercraft simulator described herein against defined criteria and reference data.

---

## Final Authoring Checks during Drafting (Per Plan 2):

*   **Parameter Table:** Ensure it is comprehensive, accurate, and all parameters have clear provenance.
*   **Flowcharts/Pseudocode/Diagrams:** Create clear visuals for key algorithms (thrust, lift, buoyancy) and structures (N-tile layout).
*   **Terminology:** Maintain rigorous consistency with definitions (use a personal glossary).
*   **Conciseness:** Be precise and avoid unnecessary jargon, especially in non-core sections. Use bullet points effectively for lists of features/details.
*   **Validation Hooks:** Ensure they are present where appropriate and point specifically to relevant sections in Chapter 4/5.
*   **Version Info/Reproducibility:** Double-check all software versions, plugin versions, and commit hashes/tags are recorded accurately. Ensure reproducibility notes are clear.
*   **Justifications:** Ensure all significant implementation choices, simplifications, and omissions are explicitly justified.

---