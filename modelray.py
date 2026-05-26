import argparse
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import ray
import xgboost as xgb

warnings.filterwarnings("ignore")

TARGET = "MUERTOS"

COLS_DESCARTAR = [
    "ID_ENTIDAD", "ID_MUNICIPIO", "ID_ACCIDENTE",
    "HERIDOS", "NEMUERTO", "NEHERIDO",
    "OTROMUERTO", "OTROHERIDO",
]

COLS_NUMERICAS = [
    "ANO", "MES", "ID_HORA", "ID_MINUTO",
    "CAMPASAJ", "MICROBUS", "PASCAMON", "OMNIBUS",
    "TRANVIA", "CAMIONETA", "CAMION", "TRACTOR",
    "FERROCARRI", "MOTOCICLET", "BICICLETA", "OTROVEHIC",
]

COLS_CATEGORICAS = [
    "DIASEMANA", "URBANA", "SUBURBANA", "TIPACCID",
    "CAUSAACCI", "CAPAROD", "SEXO", "ALIENTO",
    "CINTURON", "ID_EDAD", "CONDMANEJO", "CONDMNEJID",
    "PASAMHRID", "PEATMURID", "PEATHERID", "CICLMURID",
    "CICLHERID", "LUGAR",
]

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        7,
    "eta":              0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "alpha":            0.1,
    "lambda":           1.0,
    "seed":             42,
    "nthread":          4,
}

NUM_BOOST_ROUND = 500
NUM_WORKERS     = 4


def cargar_csv(ruta):
    for enc in ["latin-1", "cp1252", "utf-8", "iso-8859-1"]:
        try:
            df = pd.read_csv(ruta, encoding=enc, low_memory=False)
            print(f"  CSV cargado con encoding '{enc}' — {len(df):,} filas, {df.shape[1]} columnas")
            return df
        except Exception:
            continue
    raise ValueError(f"No se pudo leer: {ruta}")


def limpiar_datos(df):
    print("\n[1] Limpieza y preprocesamiento...")

    df.columns = (
        df.columns.str.strip().str.upper().str.replace(" ", "_")
          .str.replace(r"[áÁ]", "A", regex=True)
          .str.replace(r"[éÉ]", "E", regex=True)
          .str.replace(r"[íÍ]", "I", regex=True)
          .str.replace(r"[óÓ]", "O", regex=True)
          .str.replace(r"[úÚ]", "U", regex=True)
          .str.replace(r"[ñÑ]", "N", regex=True)
    )

    if TARGET not in df.columns:
        candidatas = [c for c in df.columns if "MUERT" in c]
        if candidatas:
            df.rename(columns={candidatas[0]: TARGET}, inplace=True)
            print(f"  Target renombrado: '{candidatas[0]}' -> '{TARGET}'")
        else:
            raise KeyError(f"Columna '{TARGET}' no encontrada. Columnas disponibles: {list(df.columns)}")

    cols_drop = [c for c in COLS_DESCARTAR if c in df.columns]
    df.drop(columns=cols_drop, inplace=True)
    print(f"  Columnas descartadas: {cols_drop}")

    if "COBERTURA" in df.columns:
        df["COBERTURA"] = df["COBERTURA"].apply(
            lambda x: 0 if (pd.isna(x) or str(x).strip() == "") else 1
        ).astype(int)
        print("  COBERTURA binarizada (0=sin cobertura, 1=con cobertura)")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    print(f"  Distribucion target:\n{df[TARGET].value_counts().sort_index().to_string()}")

    for col in COLS_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    le = LabelEncoder()
    for col in COLS_CATEGORICAS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str).str.strip().fillna("DESCONOCIDO"))

    for col in df.select_dtypes(include="object").columns:
        if col != TARGET:
            df[col] = le.fit_transform(df[col].astype(str).str.strip().fillna("DESCONOCIDO"))

    df.dropna(subset=[TARGET], inplace=True)
    print(f"  Filas finales: {len(df):,}")
    return df


@ray.remote
def train_fold(X_tr, y_tr, X_va, y_va, params, num_round):
    # Ray automatically deserializes ObjectRefs passed as arguments.
    # Do NOT call ray.get() here — the arrays arrive ready to use.

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        evals_result=evals_result,
        verbose_eval=100,
    )
    return booster, evals_result["valid"]["rmse"][-1]



def entrenar_con_ray(df):
    print("\n[2] Inicializando Ray...")
    ray.init(ignore_reinit_error=True)

    features = [c for c in df.columns if c != TARGET]
    X = df[features].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=np.where(y > 0, 1, 0)
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(features)}")

    # Put test set in the object store once — shared across all workers
    X_val_ref = ray.put(X_test)
    y_val_ref = ray.put(y_test)

    chunk = len(X_train) // NUM_WORKERS
    futures = []

    print(f"\n[3] Lanzando {NUM_WORKERS} workers Ray en paralelo...")
    for i in range(NUM_WORKERS):
        start = i * chunk
        end   = (i + 1) * chunk if i < NUM_WORKERS - 1 else len(X_train)

        X_chunk_ref = ray.put(X_train[start:end])
        y_chunk_ref = ray.put(y_train[start:end])

        # Pass ObjectRefs as normal arguments — Ray resolves them automatically
        futures.append(
            train_fold.remote(X_chunk_ref, y_chunk_ref, X_val_ref, y_val_ref, XGB_PARAMS, NUM_BOOST_ROUND)
        )

    results = ray.get(futures)

    best_booster, best_rmse = min(results, key=lambda r: r[1])
    print(f"\n  Entrenamiento completado.")
    print(f"  Mejor RMSE en validacion (worker): {best_rmse:.4f}")

    out_dir = "/home/lenovo/Escritorio/computo_distribuido"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.ubj")
    best_booster.save_model(model_path)
    print(f"  Modelo guardado en: {model_path}")

    return best_booster, X_test, y_test, features


def evaluar(booster, X_test, y_test, features):
    print("\n[4] Evaluando el modelo en el conjunto de test completo...")

    dtest  = xgb.DMatrix(X_test)
    y_pred = booster.predict(dtest)
    y_pred_int = np.clip(np.round(y_pred).astype(int), 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("  METRICAS " )
    print("="*50)
    print(f"  MAE  (Error Absoluto Medio)  : {mae:.4f}")
    print(f"  RMSE (Raiz Error Cuadratico) : {rmse:.4f}")
    print(f"  R2   (Coef. determinacion)   : {r2:.4f}")
    print("="*50)

    print("\n[5] Importancia de variables (Top 15)...")
    importance = booster.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(importance.items(), columns=["Feature", "Gain"])
          .sort_values("Gain", ascending=False)
          .head(15)
    )
    imp_df["Feature"] = imp_df["Feature"].apply(
        lambda f: features[int(f[1:])] if f.startswith("f") and f[1:].isdigit() else f
    )
    print(imp_df.to_string(index=False))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df["Feature"][::-1], imp_df["Gain"][::-1], color="steelblue")
        ax.set_xlabel("Gain")
        ax.set_title("Top 15 Variables - Importancia para predecir MUERTOS")
        plt.tight_layout()
        out_img = "/home/lenovo/Escritorio/computo_distribuido/importancia_variables.png"
        plt.savefig(out_img, dpi=150)
        print(f"\n  Grafica guardada en: {out_img}")
    except Exception as e:
        print(f"  No se pudo generar la grafica: {e}")

    out_pred = "/home/lenovo/Escritorio/computo_distribuido/predicciones_test.csv"
    pd.DataFrame({
        "MUERTOS_REAL":     y_test.astype(int),
        "MUERTOS_PREDICHO": y_pred_int,
    }).to_csv(out_pred, index=False)
    print(f"  Predicciones guardadas en: {out_pred}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="chihuahua_2015_2024.csv")
    args = parser.parse_args()

    df = cargar_csv(args.csv)
    df = limpiar_datos(df)
    booster, X_test, y_test, features = entrenar_con_ray(df)
    evaluar(booster, X_test, y_test, features)

    print("\nPipeline completado.")
    ray.shutdown()


if __name__ == "__main__":
    main()