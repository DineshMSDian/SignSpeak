"""
TFLite Model Converter
──────────────────────
Converts SignSpeak .keras LSTM models to .tflite format
for use in the Flutter mobile app.

Usage:
    python scripts/convert_to_tflite.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf


def convert_model(keras_path: str, tflite_path: str):
    """Convert a .keras model to .tflite format using concrete functions."""
    if not os.path.exists(keras_path):
        print(f"  [SKIP] Model not found: {keras_path}")
        return False

    print(f"  Loading: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    # Get a concrete function with fixed input signature
    input_shape = model.input_shape  # (None, 60, 126)
    run = tf.function(lambda x: model(x))
    concrete_func = run.get_concrete_function(
        tf.TensorSpec([1, input_shape[1], input_shape[2]], tf.float32)
    )

    # Convert using the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(keras_path) / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"  Saved:   {tflite_path}")
    print(f"  Size:    {original_size:.0f} KB → {tflite_size:.0f} KB")

    # Verify
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"  Input:   {inp[0]['shape']} {inp[0]['dtype']}")
    print(f"  Output:  {out[0]['shape']} {out[0]['dtype']}")

    # Sanity check: run inference
    test_input = np.random.rand(1, input_shape[1], input_shape[2]).astype(np.float32)
    interpreter.set_tensor(inp[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out[0]['index'])
    print(f"  Test OK: output shape {output.shape}, sum={output.sum():.4f}")

    return True


def main():
    print("=" * 60)
    print("  SignSpeak — TFLite Model Converter")
    print("=" * 60)

    flutter_assets = os.path.join(config.PROJECT_ROOT, "signspeak_flutter", "assets", "models")
    os.makedirs(flutter_assets, exist_ok=True)

    conversions = [
        (config.ASL_MODEL_PATH, os.path.join(flutter_assets, "sign_lstm_asl.tflite")),
        (config.ISL_MODEL_PATH, os.path.join(flutter_assets, "sign_lstm_isl.tflite")),
    ]

    label_maps = [
        (config.ASL_LABEL_MAP_PATH, os.path.join(flutter_assets, "label_map_asl.json")),
        (config.ISL_LABEL_MAP_PATH, os.path.join(flutter_assets, "label_map_isl.json")),
    ]

    print("\n── Converting Models ─────────────────────────")
    converted = 0
    for keras_path, tflite_path in conversions:
        print(f"\n  [{os.path.basename(keras_path)}]")
        if convert_model(keras_path, tflite_path):
            converted += 1

    print("\n── Copying Label Maps ────────────────────────")
    for src, dst in label_maps:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {os.path.basename(src)} → {dst}")
        else:
            print(f"  [SKIP] {src} not found")

    print(f"\n{'=' * 60}")
    print(f"  Converted {converted} model(s) to TFLite")
    print(f"  Assets saved to: {flutter_assets}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
