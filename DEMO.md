> *This document was created by Claude (Anthropic's AI assistant).*

# Open Lab Demo — `live_demo.py`

## Purpose

`live_demo.py` is the entry point for the dual-arm **cube-passing demo** on two UR5e robots with OnRobot 2FG7 grippers. The right arm picks up cubes from the table, brings each one to a shared meeting point above the table, hands it over to the left arm, and the left arm puts it down.

The script ties together the rest of the project:

| Module | Role |
|---|---|
| `Experiment.py` | Plans the full experiment (meeting point, pick-up, hand-over, put-down) and writes it to JSON |
| `planners.py`, `RRTTree.py`, `building_blocks.py` | RRT* motion planning and collision checking |
| `kinematics.py`, `inverse_kinematics.py` | UR5e forward/inverse kinematics and arm placement transforms |
| `environment.py` | Table, obstacles, cubes and arm base locations |
| `visualizer.py` | 3D matplotlib visualization of the two arms |
| `Control_robot.py`, `robot_interface.py`, `twofg7_gripper.py` | Real-robot control over RTDE and gripper control over XML-RPC |

## Functions

`live_demo.py` has four demo functions. Choose which one runs by uncommenting it in the `if __name__ == '__main__':` block at the bottom of the file:

```python
if __name__ == '__main__':
    # draw_two_robots()
    # connect_move_robots()
    # run_json("plan.json")
    create_json()
```

### 1. `create_json()` — plan the experiment (offline, no robots needed)

Runs `Experiment().plan_experiment()`, which:

1. Computes a meeting point 0.5 m above the table and finds inverse-kinematics solutions for both arms there.
2. For each cube, plans every stage with RRT* (right arm to cube, pick up, to meeting point, left arm to meeting point, hand-over, put-down).
3. Writes the plan to `outputs/plan.json` and logs the steps to `outputs/output.txt`.

The cube layout is set by `exp_id` in `Experiment.plan_experiment()`: `1` uses one cube and `2` uses two cubes. This is the default when you run the script.

### 2. `draw_two_robots()` — visualize (offline, no robots needed)

Opens a 3D matplotlib window that shows both arms in their hand-over (meeting) configurations. Use it to sanity-check the environment and arm placement.

### 3. `connect_move_robots()` — basic hardware test

Connects to both arms, sends them home, then moves the **right arm** through two hard-coded joint configurations. Use it to confirm that the connection and motion work before you run a full plan.

### 4. `run_json(json_path)` — execute a plan on the real robots

Connects to both arms, homes them, and then replays a plan file step by step. For each step it:

- selects the active arm (`active_id`),
- applies the pre-motion gripper state (`gripper_pre`: open/close),
- runs the motion — `move` follows a joint-space path and `movel` makes a relative Cartesian move,
- applies the post-motion gripper state (`gripper_post`).

## Typical workflow

1. **Plan:** run with `create_json()` enabled → produces `outputs/plan.json`.
2. **Check:** optionally run `draw_two_robots()` to view the setup.
3. **Test hardware:** run `connect_move_robots()` with a clear workspace.
4. **Execute:** run `run_json(...)` with the plan you want.

> **Note:** `create_json()` writes to `outputs/plan.json`, but `run_json("plan.json")` reads the `plan.json` in the project root. To execute a newly generated plan, either call `run_json("outputs/plan.json")` or copy the file to the root.

Run the script from the project folder, because output paths are relative (`./outputs/`):

```bash
cd "Open Lab demo"
python live_demo.py
```

## Requirements

- Python 3.12 (the version the project was last run with)
- Packages: `numpy`, `matplotlib`, `imageio`, `ur_rtde` (provides `rtde_control` / `rtde_receive`), `pycurl`

```bash
pip install numpy matplotlib imageio ur_rtde pycurl
```

Only the hardware functions (`connect_move_robots`, `run_json`) need `ur_rtde` and network access to the robots.

## Hardware configuration

The robot IP addresses are hard-coded in `connect_move_robots()` and `run_json()`:

```python
left_arm_ip = "192.168.0.17"
right_arm_ip = "192.168.0.10"
```

Change them to match your setup. Your computer must be on the same network as the robots, and both robots must be in remote-control mode. If a robot can't be reached, `BaseRobot` retries every 5 seconds until it connects.

## Safety

- Keep the emergency stop within reach whenever you run `connect_move_robots()` or `run_json()`.
- Make sure the workspace is clear: both arms move to home (`[0, -90°, 0, -90°, 0, 0]`) immediately after they connect.
- Motions block until complete; the default speed and acceleration are 0.5.
