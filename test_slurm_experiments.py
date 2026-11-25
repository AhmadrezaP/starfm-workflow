import os
from pathlib import Path
import shlex
import subprocess
import unittest

from run_nearest_pair_pipeline import build_parser as pipeline_parser
from run_modis_viirs_comparison import build_parser as sensor_parser, _pipeline_command as sensor_command
from run_search_distance_sweep import build_parser as sweep_parser, _pipeline_command as sweep_command
from run_uncertainty_sweep import build_parser as uncertainty_parser, _pipeline_command as uncertainty_command


class SlurmCommandsTests(unittest.TestCase):
    def test_generated_commands_and_linux_executable_forwarding(self):
        root = Path(__file__).resolve().parent
        parsers = {
            "run_nearest_pair_pipeline.py": pipeline_parser,
            "run_modis_viirs_comparison.py": sensor_parser,
            "run_search_distance_sweep.py": sweep_parser,
            "run_uncertainty_sweep.py": uncertainty_parser,
        }
        for mode, count in [("parameters", 2), ("sensors", 1), ("coregistration", 3), ("pairs", 4)]:
            for band in [("B3" if mode == "sensors" else "B1"), "all"]:
                for cleanup in ["0", "1"]:
                    with self.subTest(mode=mode, band=band, cleanup=cleanup):
                        result = subprocess.run(
                            ["bash", "slurm/run_experiment.sh", mode], cwd=root,
                            env={**os.environ, "DRY_RUN": "1", "BAND": band,
                                 "REMOVE_UNNECESSARY": cleanup, "EXPERIMENT_ID": "check"},
                            text=True, capture_output=True, check=True,
                        )
                        commands = [shlex.split(line[len("Running:"):]) for line in result.stdout.splitlines()
                                    if line.startswith("Running:")]
                        self.assertEqual(len(commands), count)
                        for command in commands:
                            args = parsers[command[2]]().parse_args(command[3:])
                            self.assertEqual(args.max_cloud_percent, 5)
                            self.assertEqual(args.path, 29)
                            self.assertEqual(str(args.starfm_executable), "StarFM_original/source/StarFM.exe")
                            if mode == "parameters":
                                nested = (
                                    sweep_command(args, "B1", 550, "check")
                                    if command[2] == "run_search_distance_sweep.py"
                                    else uncertainty_command(args, "B1", 20, "check")
                                )
                            elif mode == "sensors":
                                nested = sensor_command(args, "MODIS", "B3", "check")
                            else:
                                continue
                            self.assertEqual(nested[nested.index("--starfm-executable") + 1],
                                             "StarFM_original/source/StarFM.exe")

    def test_parameter_resume_is_forwarded_to_both_sweeps(self):
        root = Path(__file__).resolve().parent
        result = subprocess.run(
            ["bash", "slurm/run_experiment.sh", "parameters"], cwd=root,
            env={**os.environ, "DRY_RUN": "1", "BAND": "all", "RESUME": "1",
                 "EXPERIMENT_ID": "existing"},
            text=True, capture_output=True, check=True,
        )
        commands = [shlex.split(line[len("Running:"):]) for line in result.stdout.splitlines()
                    if line.startswith("Running:")]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--resume" in command for command in commands))
