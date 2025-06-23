
Below is the fully patched exporter; it keeps every earlier improvement but swaps in `vec.dot()` calls, so no more attribute errors.

---

### Run in **Developer Tools ▸ Output Log**

```python
import unreal, json, os, math

#──────────────────────── CONFIG ────────────────────────
PREFIXES = ("Sensor_", "gps")          # component names start with these
OUT_NAME = "sensor_mounts.txt"         # JSON wrapped in .txt
#────────────────────────────────────────────────────────

# ---------- helpers ----------
def normalize(v: unreal.Vector) -> unreal.Vector:
    mag = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
    return unreal.Vector(v.x/mag, v.y/mag, v.z/mag) if mag else unreal.Vector()

def to_body(w_vec, fwd, right, up):
    """Project world-vec into body X (fwd), Y (stbd), Z (down); metres, 4 dp."""
    return [
        round(w_vec.dot(fwd),   4),            # +F / −A
        round(w_vec.dot(right), 4),            # +S / −P
        round(-w_vec.dot(up),   4)             # +D / −U
    ]

def make_entry(comp, craft_loc, fwd, right, up):
    w_tf   = comp.get_world_transform()
    loc_cm = w_tf.translation
    delta_w = (loc_cm - craft_loc) / 100.0           # metres

    pos_b  = to_body(delta_w, fwd, right, up)

    quat   = w_tf.rotation
    x_dir  = to_body(quat.rotate_vector(unreal.Vector(1,0,0)), fwd, right, up)
    y_dir  = to_body(quat.rotate_vector(unreal.Vector(0,1,0)), fwd, right, up)
    z_tmp  = to_body(quat.rotate_vector(unreal.Vector(0,0,1)), fwd, right, up)
    z_dir  = [-v for v in z_tmp]                     # flip to +DOWN

    return {
        "device_name": comp.get_name().lower(),
        "position":    dict(zip(("x","y","z"), pos_b)),
        "x_direction": x_dir,
        "y_direction": y_dir,
        "z_direction": z_dir
    }

# ---------- main ----------
def main():
    subsys   = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    sel      = subsys.get_selected_level_actors()
    if not sel:
        unreal.log_error("Select your BP_2000TDCraft actor, then run.")
        return

    craft     = sel[0]
    craft_loc = craft.get_actor_location()            # world cm
    fwd, right, up = map(normalize, (
        craft.get_actor_forward_vector(),
        craft.get_actor_right_vector(),
        craft.get_actor_up_vector()
    ))

    entries=[]
    for comp in craft.get_components_by_class(unreal.SceneComponent):
        if any(comp.get_name().startswith(p) for p in PREFIXES):
            unreal.log(f"  ✓ found {comp.get_name()}")
            entries.append(make_entry(comp, craft_loc, fwd, right, up))

    if not entries:
        unreal.log_error("🚫  No Sensor_* or gps components found.")
        return

    path = os.path.join(unreal.Paths.project_saved_dir(), OUT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    unreal.log(f"✅  Exported {len(entries)} sensor entries → {path}")

# ---------- run ----------
if __name__ == "__main__":
    main()
```

---

### Expected outcome

* Console prints a “✓ found …” line for each sensor component.
* Finishes with
  `✅ Exported 6 sensor entries → …/Saved/sensor_mounts.txt`
* Each `position.x` for a cube 0.30 m forward of CG will read `0.3` (sign depends on your real forward axis).
* File structure matches the JSON your orientation pipeline expects.

You now have **craft-local, body-frame-correct sensor geometry** ready to drop into `frame_definitions.py` and the rest of your data-fusion workflow.