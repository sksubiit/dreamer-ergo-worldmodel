"""Small standard-library tests; all input/output lives in temporary directories."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from jsonl_to_tensorboard import Converter


def read(directory):
    acc=EventAccumulator(str(directory),size_guidance={"tensors":0}).Reload()
    return {tag:[(e.step,e.tensor_proto.double_val[0]) for e in acc.Tensors(tag)]
            for tag in acc.Tags()["tensors"]}


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/"run"
        self.run.mkdir()
        self.source=self.run/"metrics.jsonl"
        self.output=self.root/"events"

    def convert(self,follow=False):
        c=Converter(self.run,self.output,follow)
        self.addCleanup(c.close)
        c.poll(); c.flush()
        return c

    def test_sparse_nonfinite_invalid_steps_and_unfinished_tail(self):
        rows=[dict(step=17,**{"episode/score":3.5,"nan":float("nan"),"inf":float("inf"),
                              "negative_inf":-float("inf"),"big":1e100,"text":"x","missing":None,
                              "bool":True,"list":[1]}),
              dict(step=42.,**{"train/model_loss":10.,"fps":200.}),
              {"fps":300.},{"step":True,"fps":300.},{"step":"43","fps":300.},
              {"step":1.5,"fps":300.},{"step":float("nan"),"fps":300.},
              {"step":2**63,"fps":300.}]
        self.source.write_text("".join(json.dumps(r)+"\n" for r in rows)+'{"step":50,"fps":')
        c=self.convert()
        self.assertEqual(read(self.output/"segment_000"),{
            "episode/score":[(17,3.5)],"big":[(17,1e100)],
            "train/model_loss":[(42,10.)],"fps":[(42,200.)]})
        self.assertEqual(c.counts["nonfinite_values"],3)
        self.assertEqual(c.counts["invalid_steps"],6)
        self.assertGreater(c.pending_bytes,0)

    def test_complete_last_line_without_newline(self):
        self.source.write_text('{"step":91,"episode/length":32}')
        self.convert()
        self.assertEqual(read(self.output/"segment_000"),{"episode/length":[(91,32.)]})

    def test_duplicate_sparse_conflict_and_decreasing_step(self):
        rows=[{"step":10,"a":1},{"step":10,"a":1},{"step":10,"b":2},
              {"step":10,"a":3},{"step":20,"a":4},{"step":5,"a":5}]
        self.source.write_text("".join(json.dumps(r)+"\n" for r in rows))
        c=self.convert()
        self.assertEqual(c.counts["duplicate_scalars"],1)
        self.assertEqual(c.counts["conflicting_records"],1)
        self.assertEqual(len(c.segments),3)
        self.assertEqual(read(self.output/"segment_000"),{"a":[(10,1.)],"b":[(10,2.)]})
        self.assertEqual(read(self.output/"segment_001"),{"a":[(10,3.),(20,4.)]})
        self.assertEqual(read(self.output/"segment_002"),{"a":[(5,5.)]})

    def test_follow_waits_for_file_newline_and_appends(self):
        c=self.convert(follow=True)
        self.source.write_text('{"step":7,"episode/score":2')
        c.poll(); c.flush()
        self.assertEqual(c.offset,0)
        self.assertFalse(list(self.output.rglob("*tfevents*")))
        with self.source.open("a") as f:
            f.write('}\n{"step":7,"train/model_loss":3}\n')
        c.poll(); c.flush()
        c.poll(); c.flush()
        self.assertEqual(c.counts["scalars"],2)
        with self.source.open("a") as f:
            f.write('{"step":12,"fps":20}\n')
        c.poll(); c.flush()
        self.assertEqual(read(self.output/"segment_000"),{
            "episode/score":[(7,2.)],"train/model_loss":[(7,3.)],"fps":[(12,20.)]})

    def test_rewritten_file_is_a_new_segment(self):
        self.source.write_text('{"step":10,"a":1}\n')
        c=self.convert(follow=True)
        self.source.write_text('{"step":100,"a":9999999999999999}\n')
        c.poll(); c.flush()
        self.assertEqual(len(c.segments),2)
        self.assertIn("rewritten",c.segments[-1]["reason"])
        self.assertEqual(read(self.output/"segment_000"),{"a":[(10,1.)]})
        self.assertEqual(read(self.output/"segment_001")["a"][0][0],100)

    def test_malformed_complete_lines_are_skipped(self):
        self.source.write_bytes(b'not-json\n\xff\n[]\n{"step":4,"fps":9}\n')
        c=self.convert()
        self.assertEqual(c.counts["invalid_json_lines"],2)
        self.assertEqual(c.counts["invalid_records"],1)
        self.assertEqual(read(self.output/"segment_000"),{"fps":[(4,9.)]})

    def test_canonical_metrics_excludes_scores_duplicate(self):
        self.source.write_text('{"step":3,"episode/score":5}\n')
        (self.run/"scores.jsonl").write_text('{"step":3,"episode/score":999}\n')
        self.convert()
        self.assertEqual(read(self.output/"segment_000"),{"episode/score":[(3,5.)]})

    def test_output_protection_and_scores_only_input(self):
        self.source.write_text('{"step":3,"a":1}\n')
        with self.assertRaises(ValueError):
            Converter(self.run,self.run/"events")
        c=self.convert(); c.close()
        with self.assertRaises(FileExistsError):
            Converter(self.run,self.output)
        scores=self.run/"scores.jsonl"
        scores.write_text('{"step":8,"episode/score":12}\n')
        with_cleanup=Converter(scores,self.root/"score_events")
        try:
            with_cleanup.poll(); with_cleanup.flush()
            self.assertEqual(read(self.root/"score_events/segment_000"),{"episode/score":[(8,12.)]})
        finally:
            with_cleanup.close()

    def test_cli_follow_ctrl_c_flushes(self):
        script=Path(__file__).with_name("jsonl_to_tensorboard.py")
        proc=subprocess.Popen([sys.executable,"-B","-u",str(script),str(self.run),
            "--output",str(self.output),"--follow","--poll-seconds",".05"],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+10
            while not (self.output/"conversion.json").exists():
                if proc.poll() is not None or time.monotonic()>deadline:
                    self.fail("Follow process did not start")
                time.sleep(.02)
            self.source.write_text('{"step":125,"results/success":0}\n')
            while True:
                try:
                    report=json.loads((self.output/"conversion.json").read_text())
                except FileNotFoundError:
                    report={}
                if report.get("counts",{}).get("scalars")==1:
                    break
                if proc.poll() is not None or time.monotonic()>deadline:
                    self.fail("Follow process did not consume appended record")
                time.sleep(.02)
            proc.send_signal(signal.SIGINT)
            stdout,stderr=proc.communicate(timeout=5)
            self.assertEqual(proc.returncode,0,(stdout,stderr))
            self.assertIn("flushing",stdout)
            self.assertEqual(read(self.output/"segment_000"),{"results/success":[(125,0.)]})
        finally:
            if proc.poll() is None:
                proc.kill(); proc.communicate(timeout=5)


if __name__=="__main__":
    unittest.main()
