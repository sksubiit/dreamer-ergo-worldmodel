"""Standalone scalar conversion; does not import Dreamer, JAX or TensorFlow.

Use metrics.jsonl as canonical input; scores.jsonl is its redundant score export.
Each invocation requires a fresh output directory. A step decrease or conflicting
value at a repeated step starts a separate TensorBoard segment, without changing
steps or discarding the earlier history.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class Converter:
    def __init__(self, source, output, follow=False):
        source=Path(source).resolve()
        self.source=source if source.suffix==".jsonl" else source/"metrics.jsonl"
        self.output=Path(output).resolve()
        if self.output==self.source.parent or self.source.parent in self.output.parents:
            raise ValueError("Output must be outside the original log directory")
        if not follow and not self.source.is_file():
            raise FileNotFoundError(self.source)
        self.output.mkdir(parents=True,exist_ok=False)
        self.follow=follow
        self.offset=0
        self.identity=None
        self.anchor=b""
        self.writer=None
        self.segment=0
        self.last_step=None
        self.seen={}
        self.tags=set()
        self.segments=[dict(directory="segment_000",reason="initial",first_step=None,last_step=None)]
        self.counts=Counter()
        self.pending_bytes=0

    def warn(self, message):
        print(f"Warning: {message}",file=sys.stderr,flush=True)

    def new_segment(self, reason):
        if self.writer:
            self.writer.close()
            self.writer=None
        self.segment+=1
        self.last_step=None
        self.seen={}
        directory=f"segment_{self.segment:03d}"
        self.segments.append(dict(directory=directory,reason=reason,first_step=None,last_step=None))
        self.counts["segment_boundaries"]+=1
        self.warn(f"{reason}; starting {directory} (original steps retained)")

    def record(self, row):
        if not isinstance(row,dict):
            self.counts["invalid_records"]+=1
            self.warn("Skipping a JSON value that is not an object")
            return
        step=row.get("step")
        if (type(step) not in (int,float) or
            isinstance(step,float) and (not math.isfinite(step) or not step.is_integer()) or
            not 0<=step<=2**63-1):
            self.counts["invalid_steps"]+=1
            self.warn("Skipping record without a nonnegative integer int64 step")
            return
        step=int(step)
        scalars={}
        for tag,value in row.items():
            if tag=="step":
                continue
            if type(value) not in (int,float):
                self.counts["nonnumeric_values"]+=1
                continue
            try:
                value=float(value)
            except OverflowError:
                self.counts["nonfinite_values"]+=1
                continue
            if not math.isfinite(value):
                self.counts["nonfinite_values"]+=1
                continue
            scalars[tag]=value
        if self.last_step is not None and step<self.last_step:
            self.new_segment(f"Step decreased from {self.last_step} to {step}")
        elif step==self.last_step and any(k in self.seen and self.seen[k]!=v for k,v in scalars.items()):
            self.counts["conflicting_records"]+=1
            self.new_segment(f"Conflicting scalar value at repeated step {step}")
        if step!=self.last_step:
            self.seen={}
        self.last_step=step
        segment=self.segments[-1]
        if segment["first_step"] is None:
            segment["first_step"]=step
        segment["last_step"]=step
        values=[]
        for tag,value in scalars.items():
            if tag in self.seen:
                self.counts["duplicate_scalars"]+=1
                continue
            self.seen[tag]=value
            self.tags.add(tag)
            # Double tensors avoid simple_value's float32 overflow/precision loss.
            values.append(summary_pb2.Summary.Value(tag=tag,
                metadata=summary_pb2.SummaryMetadata(plugin_data=
                    summary_pb2.SummaryMetadata.PluginData(plugin_name="scalars")),
                tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_DOUBLE,double_val=[value])))
        if values:
            if self.writer is None:
                self.writer=EventFileWriter(str(self.output/segment["directory"]),flush_secs=1)
            self.writer.add_event(event_pb2.Event(wall_time=time.time(),step=step,
                summary=summary_pb2.Summary(value=values)))
            self.counts["scalars"]+=len(values)
        self.counts["records"]+=1

    def poll(self):
        """Consume complete lines; follow mode keeps an unterminated tail pending."""
        try:
            stream=self.source.open("rb")
        except FileNotFoundError:
            if self.follow:
                return
            raise
        with stream:
            stat=os.fstat(stream.fileno())
            identity=(stat.st_dev,stat.st_ino)
            changed=self.identity is not None and identity!=self.identity
            if self.offset:
                stream.seek(self.offset-len(self.anchor))
                changed=changed or stat.st_size<self.offset or stream.read(len(self.anchor))!=self.anchor
            if changed:
                self.new_segment("Input replaced, truncated or rewritten")
                self.offset=0
                self.anchor=b""
            self.identity=identity
            stream.seek(self.offset)
            self.pending_bytes=0
            while True:
                line=stream.readline()
                if not line:
                    break
                complete=line.endswith(b"\n")
                if not complete and self.follow:
                    self.pending_bytes=len(line)
                    break
                try:
                    row=json.loads(line.decode("utf-8"))
                except (ValueError,UnicodeError):
                    if not complete:
                        self.pending_bytes=len(line)
                        self.warn("Ignoring incomplete final JSONL line; append it before a fresh conversion")
                        break
                    self.counts["invalid_json_lines"]+=1
                    self.warn(f"Skipping invalid JSONL line at byte {self.offset}")
                else:
                    self.record(row)
                self.offset=stream.tell()
            # Verify this committed tail on the next poll to detect same-inode rewrites.
            stream.seek(max(0,self.offset-128))
            self.anchor=stream.read(min(128,self.offset))

    def flush(self):
        if self.writer:
            self.writer.flush()
        report=dict(source=str(self.source),counts=dict(self.counts),segments=self.segments,
                    tags=sorted(self.tags),committed_bytes=self.offset,pending_tail_bytes=self.pending_bytes,
                    wall_time="Conversion time; JSONL has no original event timestamps",
                    restart_policy="Separate segment on decreasing steps, conflicting repeated steps, or file rewrite",
                    source_policy="One canonical input only; metrics.jsonl includes scores.jsonl exports")
        temporary=self.output/"conversion.json.tmp"
        temporary.write_text(json.dumps(report,indent=2)+"\n")
        temporary.replace(self.output/"conversion.json")
        return report

    def close(self):
        report=self.flush()
        if self.writer:
            self.writer.close()
            self.writer=None
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="Run directory or a single JSONL file")
    parser.add_argument("--output",type=Path,required=True,help="Fresh directory outside the run")
    parser.add_argument("--follow",action="store_true",help="Wait for the file and appended newline-terminated records")
    parser.add_argument("--poll-seconds",type=float,default=1.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds<=0:
        parser.error("--poll-seconds must be finite and positive")
    converter=None
    try:
        converter=Converter(args.input,args.output,args.follow)
        print(f"Reading {converter.source}; writing {converter.output}",flush=True)
        if args.follow:
            print("Following; waiting for appended complete lines. Ctrl+C flushes and exits.",flush=True)
        while True:
            converter.poll()
            converter.flush()
            if not args.follow:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopped; flushing events.",flush=True)
    except (OSError,ValueError) as exc:
        print(f"Error: {exc}. Use a fresh output directory when restarting the converter.",file=sys.stderr)
        return 2
    finally:
        if converter is not None:
            report=converter.close()
            print(json.dumps({k:report[k] for k in ("counts","segments","pending_tail_bytes")}),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
