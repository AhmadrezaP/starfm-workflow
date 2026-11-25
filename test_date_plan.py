import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from slurm.date_plan import make_tasks, read_dates, save_json, submit, submit_array, task_environment, execute, collect


class DatePlanTests(unittest.TestCase):
    def test_excel_reader_keeps_pair_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'dates.xlsx'
            with ZipFile(path, 'w') as archive:
                archive.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="path 29 - row 32" r:id="rId1"/></sheets></workbook>')
                archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
                archive.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Pair 1</t></is></c><c r="B1" t="inlineStr"><is><t>Prediction date</t></is></c><c r="C1" t="inlineStr"><is><t>Pair 2</t></is></c></row><row r="2"><c r="A2" t="d"><v>2023-09-26</v></c><c r="B2" t="d"><v>2023-09-18</v></c><c r="C2" t="d"><v>2023-09-02</v></c></row></sheetData></worksheet>')
            cases = read_dates(path)
            self.assertEqual(cases, [self.case()])

    def case(self):
        return dict(path=29, row=32, prediction_date="2023-09-18",
                    pair_dates=["2023-09-26", "2023-09-02"], source_sheet="path 29 - row 32", source_row=2)

    def plan(self, root):
        return dict(output_dir=str(root), cloud_percent=5, remove_unnecessary=True,
                    starfm_executable="StarFM_original/source/StarFM.exe", tasks=make_tasks([self.case()], "test"))

    def test_requested_matrix(self):
        tasks = make_tasks([self.case()], "test")
        self.assertEqual(len(tasks), 82)
        self.assertEqual(len({t['test_id'] for t in tasks}), 82)
        self.assertEqual({t['distance'] for t in tasks if t['family'] == 'distance'}, {550,650,750,850,950,1150})
        self.assertEqual({t['uncertainty'] for t in tasks if t['family'] == 'uncertainty'}, {35,50,65})
        self.assertFalse(any(t['band'] == 'B1' and t['family'] == 'sensors' for t in tasks))
        self.assertTrue(all(t['pair_dates'] == self.case()['pair_dates'] for t in tasks))

    def test_final_configuration(self):
        cases = [dict(self.case(), path=p, row=r) for p, r in [(29, 32), (43, 34), (143, 42)]]
        tasks = make_tasks(cases, 'final_v1', mode='final')
        self.assertEqual(len(tasks), 18)
        self.assertEqual(len({t['test_id'] for t in tasks}), 18)
        for task in tasks:
            self.assertEqual((task['method'], task['distance'], task['uncertainty']), ('moving-window', 950, 35))
            self.assertEqual(task['family'], 'final')
            self.assertEqual(task['pair_dates'], self.case()['pair_dates'])
            expected = 'VIIRS' if task['band'] == 'B1' or task['path'] == 143 else 'MODIS'
            self.assertEqual(task['sensor'], expected)
            environment = task_environment(self.plan(Path('/tmp')), task)
            self.assertEqual(environment['COREGISTRATION_BAND'], task['band'])
            self.assertEqual(environment['VALIDATE_HELD_OUT'], '1')
            self.assertEqual(environment['SEARCH_DISTANCE'], '950')
            self.assertEqual(environment['UNCERTAINTY'], '35')

    def test_final_rejects_unknown_region(self):
        with self.assertRaisesRegex(ValueError, 'sensor policy'):
            make_tasks([dict(self.case(), path=1)], 'final_v1', mode='final')

    def test_array_time_throttle_and_no_double_submit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            plan = self.plan(Path(directory))
            plan['tasks'] = plan['tasks'][:6]
            save_json(path, plan)
            with patch('slurm.date_plan.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '501\n', '')) as run:
                submit_array(path, 8)
                command = run.call_args.args[0]
                self.assertIn('--array=0-5%8', command)
                self.assertIn('--time=04:00:00', command)
                self.assertFalse(any(x.startswith('--dependency') for x in command))
                run.reset_mock()
                submit_array(path, 8)
                run.assert_not_called()

    def test_array_retry_incomplete_only_and_reject_active(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'plan.json'
            plan = self.plan(root)
            plan['tasks'] = plan['tasks'][:3]
            save_json(path, plan)
            save_json(root / 'array_submissions.json', [{'job_id': '501'}])
            completed = root / 'OUTPUTS/TESTS' / plan['tasks'][1]['test_id']
            completed.mkdir(parents=True)
            (completed / 'test_summary.json').write_text('{}')
            with patch('slurm.date_plan.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '501_0\n', '')):
                with self.assertRaisesRegex(RuntimeError, 'active'):
                    submit_array(path, 4, retry=True)
            responses = [subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 0, '502\n', '')]
            with patch('slurm.date_plan.subprocess.run', side_effect=responses) as run:
                submit_array(path, 4, retry=True)
                self.assertIn('--array=0,2%4', run.call_args.args[0])
                self.assertEqual(run.call_args.kwargs['env']['RETRY_INCOMPLETE'], '1')

    def test_array_retry_after_slurm_forgets_job_and_rejects_scheduler_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'plan.json'
            plan = self.plan(root)
            plan['tasks'] = plan['tasks'][:2]
            save_json(path, plan)
            save_json(root / 'array_submissions.json', [{'job_id': '501'}])
            unavailable = subprocess.CompletedProcess([], 1, '', 'slurm_load_jobs error: Unable to contact slurm controller\n')
            with patch('slurm.date_plan.subprocess.run', return_value=unavailable) as run:
                with self.assertRaisesRegex(RuntimeError, 'squeue failed'):
                    submit_array(path, 4, retry=True)
                self.assertEqual(run.call_count, 1)
            forgotten = subprocess.CompletedProcess([], 1, '', 'slurm_load_jobs error: Invalid job id specified\n')
            with patch('slurm.date_plan.subprocess.run', side_effect=[forgotten, subprocess.CompletedProcess([], 0, '503\n', '')]) as run:
                submit_array(path, 4, retry=True)
                self.assertIn('--array=0-1%4', run.call_args.args[0])

    def test_array_retry_can_be_limited_to_selected_tasks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'plan.json'
            plan = self.plan(root)
            plan['tasks'] = plan['tasks'][:4]
            save_json(path, plan)
            save_json(root / 'array_submissions.json', [{'job_id': '501'}])
            responses = [subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 0, '504\n', '')]
            with patch('slurm.date_plan.subprocess.run', side_effect=responses) as run:
                submit_array(path, 30, retry=True, only=[2])
                self.assertIn('--array=2%30', run.call_args.args[0])
            with self.assertRaisesRegex(ValueError, 'Unknown task'):
                submit_array(path, 30, retry=True, only=[99])

    def test_collect_merges_completed_override_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base_case = dict(self.case(), source_row=10)
            override_case = dict(self.case(), source_row=2)
            plans = {}
            for name, case, cloud in (('base', base_case, 5), ('extra', override_case, 6)):
                folder = root / name
                folder.mkdir()
                tasks = make_tasks([case], name, mode='experiments')[:3]
                plans[name] = folder / 'plan.json'
                save_json(plans[name], dict(campaign=name, output_dir=str(root), cloud_percent=cloud, tasks=tasks))
            base_tasks = json.loads(plans['base'].read_text())['tasks']
            extra_tasks = json.loads(plans['extra'].read_text())['tasks']
            for task, rmse in ((base_tasks[0], 1.0), (extra_tasks[1], 2.0)):
                folder = root / 'OUTPUTS/TESTS' / task['test_id']
                folder.mkdir(parents=True)
                (folder / 'test_summary.json').write_text(json.dumps({'rmse': rmse}))
            collect(plans['base'], [plans['extra']])
            merged = json.loads((plans['base'].parent / 'results_merged.json').read_text())
            self.assertEqual([r['status'] for r in merged], ['completed', 'completed', 'not_started'])
            self.assertEqual(merged[1]['rmse'], 2.0)
            self.assertEqual((merged[1]['campaign'], merged[1]['cloud_percent'], merged[1]['source_row']), ('extra', 6, 10))
            self.assertEqual(merged[1]['test_id'], extra_tasks[1]['test_id'])
            base_only = json.loads((plans['base'].parent / 'results.json').read_text())
            self.assertEqual(base_only[1]['status'], 'not_started')
            unrelated = root / 'other' / 'plan.json'
            unrelated.parent.mkdir()
            other = dict(self.case(), prediction_date='2023-09-20')
            save_json(unrelated, dict(campaign='other', output_dir=str(root), cloud_percent=6, tasks=make_tasks([other], 'other')[:1]))
            with self.assertRaisesRegex(ValueError, 'matches no prediction'):
                collect(plans['base'], [unrelated])

    def test_retry_preserves_incomplete_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'plan.json'
            plan = self.plan(root)
            save_json(path, plan)
            folder = root / 'OUTPUTS/TESTS' / plan['tasks'][0]['test_id']
            folder.mkdir(parents=True)
            (folder / 'partial.txt').write_text('keep')
            with patch.dict('os.environ', {'RETRY_INCOMPLETE': '1'}), patch('slurm.date_plan.subprocess.run', return_value=subprocess.CompletedProcess([], 1)):
                self.assertEqual(execute(path, 0), 1)
            backups = list(folder.parent.glob(folder.name + '.failed_*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / 'partial.txt').read_text(), 'keep')

    def test_jobs_are_serial_chained_and_submission_resume(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            plan = self.plan(Path(directory))
            plan['tasks'] = plan['tasks'][:3]
            save_json(path, plan)
            responses = [subprocess.CompletedProcess([], 0, str(i), '') for i in [101,102,103]]
            with patch('slurm.date_plan.subprocess.run', side_effect=responses) as run:
                submit(path)
            calls = run.call_args_list
            self.assertFalse(any(arg.startswith('--dependency') for arg in calls[0].args[0]))
            self.assertIn('--dependency=afterany:101', calls[1].args[0])
            self.assertIn('--dependency=afterany:102', calls[2].args[0])
            self.assertEqual(calls[2].kwargs['env']['PLAN_INDEX'], '2')
            with patch('slurm.date_plan.subprocess.run') as run:
                submit(path)
                run.assert_not_called()

    def test_environment_and_existing_summary_skip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            task = plan['tasks'][1]
            env = task_environment(plan, task)
            self.assertEqual(env['COREGISTRATION_BAND'], 'B5')
            self.assertEqual(env['PAIR_DATE_1'], '2023-09-26')
            self.assertEqual(env['INTERMEDIATE_OUTPUTS'], 'discard')
            summary = root / 'OUTPUTS/TESTS' / task['test_id'] / 'test_summary.json'
            summary.parent.mkdir(parents=True)
            save_json(summary, {'rmse_surface_reflectance': 0.1})
            path = root / 'plan.json'
            save_json(path, plan)
            with patch('slurm.date_plan.subprocess.run') as run:
                self.assertEqual(execute(path, 1), 0)
                run.assert_not_called()
            collect(path)
            results = json.loads((root / 'results.json').read_text())
            self.assertEqual(results[1]['status'], 'completed')
            self.assertEqual(results[0]['status'], 'not_started')

    def test_failure_record(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'plan.json'
            save_json(path, self.plan(root))
            with patch('slurm.date_plan.subprocess.run', return_value=subprocess.CompletedProcess([], 1)):
                self.assertEqual(execute(path, 0), 1)
            self.assertEqual(json.loads((root / 'status/00000.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
