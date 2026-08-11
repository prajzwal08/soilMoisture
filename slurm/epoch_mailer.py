#!/usr/bin/env python3
"""
Watch a training log and send an email summary after each epoch.

Usage:
    python slurm/epoch_mailer.py --log logs/train_23921632.out \
                                  --to ktm.prajwalkhanal@gmail.com \
                                  --run baseline_huber
"""
import argparse
import re
import smtplib
import subprocess
import time
from email.mime.text import MIMEText
from pathlib import Path


EPOCH_RE   = re.compile(r"Epoch (\d+)\s+\|(.+)")
DEPTH_RE   = re.compile(r"(\S+cm)\s+MSE=([\d.]+)\s+MAE=([\d.]+)\s+ubRMSE=([\d.]+)\s+bias=([-\d.]+)")
MEM_RE     = re.compile(r"RAM\s+used\s*:\s*([\d.]+)\s*GB")
VRAM_RE    = re.compile(r"GPU 0 VRAM:\s*([\d.]+)\s+alloc")
SNAP_RE    = re.compile(r"=== Memory snapshot: epoch_(\d+)_post_val")


def send_mail(to: str, subject: str, body: str):
    """Send via local sendmail (works on Snellius login/compute nodes)."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = "snellius-train@noreply"
    msg["To"]      = to
    try:
        with smtplib.SMTP("localhost") as s:
            s.sendmail("snellius-train@noreply", [to], msg.as_string())
        return True
    except Exception:
        # Fallback: pipe through mail command
        try:
            proc = subprocess.run(
                ["mail", "-s", subject, to],
                input=body.encode(), capture_output=True, timeout=10
            )
            return proc.returncode == 0
        except Exception as e:
            print(f"[mailer] send failed: {e}")
            return False


def format_email(run: str, epoch: int, summary_line: str,
                 depth_lines: list, ram_gb: str, vram_gb: str) -> tuple[str, str]:
    subject = f"[{run}] Epoch {epoch:03d} done"

    lines = [f"Run: {run}", f"Epoch: {epoch:03d}", ""]
    # parse summary line
    m = re.search(
        r"train_loss=([\d.]+).*?val_loss=([\d.]+).*?data=(\S+).*?compute=(\S+).*?gpu_util=(\S+).*?peak_vram=(\S+)",
        summary_line
    )
    if m:
        lines += [
            "=== Losses ===",
            f"  train_loss : {m.group(1)}",
            f"  val_loss   : {m.group(2)}",
            "",
            "=== Timing ===",
            f"  data load  : {m.group(3)}",
            f"  compute    : {m.group(4)}",
            f"  gpu_util   : {m.group(5)}",
            f"  peak_vram  : {m.group(6)}",
            "",
        ]

    if depth_lines:
        lines.append("=== Val Metrics per Depth ===")
        for d in depth_lines:
            dm = DEPTH_RE.search(d)
            if dm:
                lines.append(
                    f"  {dm.group(1):>8s}  ubRMSE={dm.group(4)}  MAE={dm.group(3)}"
                    f"  MSE={dm.group(2)}  bias={dm.group(5)}"
                )
        lines.append("")

    lines += [
        "=== Memory ===",
        f"  RAM used   : {ram_gb} GB",
        f"  VRAM alloc : {vram_gb} GB (GPU 0)",
        "",
        "Full log: logs/train_*.out",
    ]
    return subject, "\n".join(lines)


def watch(log_path: Path, to: str, run: str, poll: float = 5.0):
    print(f"[mailer] watching {log_path} → {to}")
    sent_epochs: set[int] = set()
    buf: list[str] = []
    in_epoch_block = False
    current_epoch  = -1
    depth_lines: list[str] = []
    ram_gb  = "?"
    vram_gb = "?"

    with open(log_path, "r") as f:
        f.seek(0, 2)   # start at end — don't replay old epochs
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll)
                continue

            # Detect epoch summary line
            m = EPOCH_RE.search(line)
            if m:
                current_epoch  = int(m.group(1))
                in_epoch_block = True
                depth_lines    = []
                buf            = [line.strip()]
                continue

            if in_epoch_block:
                dm = DEPTH_RE.search(line)
                if dm:
                    depth_lines.append(line.strip())
                    continue

                # memory snapshot post_val — grab RAM/VRAM then fire email
                if SNAP_RE.search(line):
                    # read a few more lines for RAM/VRAM
                    for _ in range(10):
                        extra = f.readline()
                        if not extra:
                            time.sleep(0.5)
                            extra = f.readline()
                        mm = MEM_RE.search(extra)
                        if mm:
                            ram_gb = mm.group(1)
                        vm = VRAM_RE.search(extra)
                        if vm:
                            vram_gb = vm.group(1)
                        if ram_gb != "?" and vram_gb != "?":
                            break

                    if current_epoch not in sent_epochs:
                        subj, body = format_email(
                            run, current_epoch,
                            buf[0] if buf else "",
                            depth_lines, ram_gb, vram_gb
                        )
                        ok = send_mail(to, subj, body)
                        if ok:
                            print(f"[mailer] sent epoch {current_epoch:03d} email")
                            sent_epochs.add(current_epoch)
                        else:
                            print(f"[mailer] email failed for epoch {current_epoch:03d}")
                    in_epoch_block = False
                    continue


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log",  required=True, help="Path to training log file")
    p.add_argument("--to",   default="ktm.prajwalkhanal@gmail.com")
    p.add_argument("--run",  default="baseline_huber")
    p.add_argument("--poll", type=float, default=5.0, help="Poll interval seconds")
    args = p.parse_args()
    watch(Path(args.log), args.to, args.run, args.poll)
