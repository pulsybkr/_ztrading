"""
Export du modele LightGBM vers ONNX pour utilisation dans un EA MT5.

Usage:
    python scripts/export_onnx.py --timeframe M5
    python scripts/export_onnx.py --timeframe M5 --output ea/model_M5.onnx

Requirements:
    pip install skl2onnx onnx
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from ml.features import FEATURE_COLUMNS
from ml.trainer import load_model


def export_to_onnx(timeframe: str = "M5", output: str = None):
    tf = timeframe.upper()
    model_path = f"ml/models/model_{tf}.joblib"

    if not Path(model_path).exists():
        print(f"Modele introuvable: {model_path}")
        print("Lance d'abord: python -m cli.app train run --timeframe M5 ...")
        return

    if output is None:
        output = f"ea/model_{tf}.onnx"

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du modele: {model_path}")
    model, metadata = load_model(model_path)

    n_features = len(FEATURE_COLUMNS)
    print(f"  Features ({n_features}): {FEATURE_COLUMNS}")
    print(f"  Precision: {metadata.get('precision', '?')}")

    try:
        import lightgbm as lgb
        from skl2onnx import to_onnx, update_registered_converter
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

        # Enregistrer le convertisseur LightGBM dans skl2onnx
        update_registered_converter(
            lgb.LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False]},
        )

        X_sample = np.zeros((1, n_features), dtype=np.float32)

        # zipmap=False -> sortie float32 [N, 2] au lieu de map (incompatible MT5)
        onnx_model = to_onnx(
            model,
            X_sample,
            options={"zipmap": False},
            target_opset={"": 12, "ai.onnx.ml": 3},
        )

        with open(output, "wb") as f:
            f.write(onnx_model.SerializeToString())

        size_kb = Path(output).stat().st_size / 1024
        print(f"\n[OK] Modele ONNX sauvegarde: {output} ({size_kb:.1f} KB)")
        print(f"\nProchaine etape:")
        print(f"  1. Copier {output} dans:")
        username = os.getenv("USERNAME", "USER")
        print(f"     C:/Users/{username}/AppData/Roaming/MetaQuotes/Terminal/<ID>/MQL5/Files/")
        print(f"  2. Copier ea/NovaGold_ML.mq5 dans MQL5/Experts/")
        print(f"  3. Compiler et lancer l'EA dans MetaEditor (F7)")

    except ImportError as e:
        print(f"[ERREUR] Dependance manquante: {e}")
        print("Lance: pip install skl2onnx onnxmltools onnx")
        return
    except Exception as e:
        print(f"[ERREUR] Export echoue: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="M5", help="Timeframe du modele: M5, M1, M15...")
    parser.add_argument("--output", default=None, help="Chemin de sortie .onnx")
    args = parser.parse_args()
    export_to_onnx(args.timeframe, args.output)
