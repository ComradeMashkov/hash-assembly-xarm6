# hash assembly, 2x xArm6

Code for the ETH technical assignment (drawing AMZ-SA-26-001). Report is `Mashkov_hash_assembly_report.pdf`.

No real robot, everything is simulation (MuJoCo) and calculation.

what is where:
- `src/tolerance_mc.py` - monte carlo for the tolerances
- `src/run_cell.py` - the cell with both robots, `--mode lock` or `--mode cbf`
- `src/capture_mc.py` - how much error the fixture can take
- `src/cad_fixture.py` - fixture in cadquery, step file is in `cad/`
- `media/` - videos, `comparison.mp4` is both modes side by side

to run (python 3.12):

```
pip install numpy scipy matplotlib mujoco imageio imageio-ffmpeg cadquery pymupdf pillow
git clone --depth 1 https://github.com/xArm-Developer/xarm_ros2 ../third_party/xarm_ros2
cd src
python tolerance_mc.py
python run_cell.py --mode cbf --video ../media/cycle_cbf.mp4
```

xArm6 meshes are from the official xarm_ros2 repo, thats why the clone.
