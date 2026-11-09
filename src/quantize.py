"""ONNX export and the quantization ladder: fp32 -> fp16 -> int8 -> int4.

Every rung is benchmarked on the same held-out images, so the report gets one honest table:
accuracy vs. size vs. latency.

    python -m src.quantize --ckpt runs/best.pt --calib data/raw/plantvillage/valid --out results/quant

Notes on precision formats
--------------------------
fp32  baseline, what PyTorch trained.
fp16  halves the file, negligible accuracy change, needs a GPU or AVX512-FP16 for speed-up.
int8  the real win on CPU phones: dynamic (weights only) or static (weights + activations,
      needs a calibration set). Static is usually 1-2% worse than fp32 and 3-4x faster.
int4  weight-only 4-bit (MatMulNBits / GPTQ-style). Tiny file, but a CNN loses several points
      of accuracy — treat it as an experiment, not a deployment target.
fp4   NOT implemented. FP4 (NVFP4 / MXFP4) needs Blackwell-class hardware and NVIDIA ModelOpt
      or TensorRT tooling; there is no ONNX Runtime path that runs on Colab or CPU. Reporting a
      number for it here would be invented, so this module refuses instead. Put it in the report
      as future work with that reason.

tflite is deliberately a separate later step (ai-edge-torch); see export_tflite() below.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import build_transforms, index_folder
from src.models.disease import build_model


# ------------------------------------------------------------------ export
def export_fp32(ckpt: Path, out: Path, size: int = 224) -> Path:
    ck = torch.load(ckpt, map_location="cpu")
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False).eval()
    model.load_state_dict(ck["model"])
    dummy = torch.randn(1, 3, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    return out


def convert_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(src))
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(dst))
    return dst


def quantize_int8_dynamic(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


class _CalibReader:
    """Feeds real validation images to the static quantizer."""

    def __init__(self, items, transform, size=224, limit=200):
        self.transform = transform
        self.size = size
        self.items = items[:limit]

    def __len__(self):
        return len(self.items)

    def get_item(self, i):
        with Image.open(self.items[i][0]) as im:
            x = self.transform(im.convert("RGB")).unsqueeze(0).numpy()
        return {"input": x.astype(np.float32)}


def quantize_int8_static(src: Path, dst: Path, calib_items, transform) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.r = _CalibReader(calib_items, transform)
            self.i = 0

        def get_next(self):
            if self.i >= len(self.r):
                return None
            batch = self.r.get_item(self.i)
            self.i += 1
            return batch

    quantize_static(str(src), str(dst), Reader(), quant_format=__import__(
        "onnxruntime.quantization", fromlist=["QuantFormat"]).QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=True)
    return dst


def quantize_int4(src: Path, dst: Path, block_size: int = 128) -> Path:
    """Weight-only 4-bit. API moved between ORT versions; try the modern one, then the legacy."""
    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
        from onnxruntime.quantization.onnx_quantizer import ONNXModel
        model = ONNXModel(__import__("onnx").load(str(src)))
        MatMul4BitsQuantizer(model, block_size=block_size, is_symmetric=True).process()
        model.save_model_to_file(str(dst), use_external_data_format=False)
        return dst
    except Exception as e:  # noqa: BLE001 - version drift is expected, we report it plainly
        raise RuntimeError(
            f"int4 weight-only quantization unavailable in this onnxruntime build ({e}). "
            "Install onnxruntime>=1.18 and retry, or skip the int4 rung."
        ) from e


# ------------------------------------------------------------------ benchmark
def benchmark(onnx_path: Path, items, transform, classes, n: int = 200, warmup: int = 5) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    items = items[:n]
    xs = []
    for p, _ in items:
        with Image.open(p) as im:
            xs.append(transform(im.convert("RGB")).unsqueeze(0).numpy().astype(np.float32))

    for x in xs[:warmup]:
        sess.run(None, {name: x})

    t0 = time.perf_counter()
    preds = []
    for x in xs:
        preds.append(int(np.argmax(sess.run(None, {name: x})[0], 1)[0]))
    dt = time.perf_counter() - t0

    labels = [y for _, y in items]
    acc = float(np.mean([p == y for p, y in zip(preds, labels)]))
    return {
        "file": onnx_path.name,
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "latency_ms_per_image": round(dt / max(len(xs), 1) * 1000, 2),
        "top1_accuracy": round(acc, 4),
        "n_images": len(xs),
    }


def export_tflite(*_args, **_kwargs):
    raise NotImplementedError(
        "tflite export is scheduled after 15 Nov. Use ai-edge-torch: "
        "pip install ai-edge-torch, then ai_edge_torch.convert(model.eval(), (sample_input,)). "
        "Verify with the TFLite interpreter and re-run the int8 calibration there — "
        "TFLite quantizes differently from ONNX Runtime, so numbers do not transfer."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True, help="validation image folder for int8 calibration")
    ap.add_argument("--out", type=Path, default=Path("results/quant"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bench-n", type=int, default=200)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    classes = ck["classes"]
    items, _ = index_folder(args.calib)
    tf = build_transforms(False, args.size)

    print("[1/5] exporting fp32")
    fp32 = export_fp32(args.ckpt, args.out / "model_fp32.onnx", args.size)

    ladder = [("fp32", fp32)]
    steps = [
        ("fp16", lambda: convert_fp16(fp32, args.out / "model_fp16.onnx")),
        ("int8_dynamic", lambda: quantize_int8_dynamic(fp32, args.out / "model_int8_dyn.onnx")),
        ("int8_static", lambda: quantize_int8_static(fp32, args.out / "model_int8_static.onnx", items, tf)),
        ("int4", lambda: quantize_int4(fp32, args.out / "model_int4.onnx")),
    ]
    for i, (name, fn) in enumerate(steps, start=2):
        print(f"[{i}/5] {name}")
        try:
            ladder.append((name, fn()))
        except Exception as e:  # noqa: BLE001
            print(f"   SKIPPED {name}: {e}")

    print("[bench]")
    rows = []
    for name, path in ladder:
        try:
            r = benchmark(path, items, tf, classes, args.bench_n)
            r["precision"] = name
            rows.append(r)
            print("  ", json.dumps(r))
        except Exception as e:  # noqa: BLE001
            print(f"   benchmark failed for {name}: {e}")

    (args.out / "quantization_report.json").write_text(json.dumps(rows, indent=2))
    md = ["| precision | size (MB) | latency (ms/img) | top-1 |", "| --- | --- | --- | --- |"]
    md += [f"| {r['precision']} | {r['size_mb']} | {r['latency_ms_per_image']} | {r['top1_accuracy']:.4f} |" for r in rows]
    md += ["", "fp4 not measured: no ONNX Runtime path without Blackwell-class hardware. See module docstring."]
    (args.out / "quantization_report.md").write_text("\n".join(md))
    print(f"\nwrote {args.out}/quantization_report.md")


if __name__ == "__main__":
    main()
