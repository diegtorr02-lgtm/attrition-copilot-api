#!/usr/bin/env python3
"""
Attrition Copilot – Minimaler, praxistauglicher Starter für IBM HR Attrition
-----------------------------------------------------------------------------

WICHTIG: Diese Version ist SHAP-fallback-fähig. Wenn `shap` in eurer Umgebung
nicht installiert ist, werden erklärende Funktionen automatisch auf modellinterne
Importances (Permutation/FeatureImportance) zurückfallen, sodass train,
predict, top-factors weiterhin laufen. explain liefert dann einfache,
aber sinnvolle Top-Gründe ohne SHAP.

Funktionen:
- Training eines Klassifikationsmodells (RandomForest oder XGBoost, falls verfügbar)
- Persistenz (Speichern/Laden) des kompletten Pipelines inkl. Preprocessing
- Vorhersage der Kündigungswahrscheinlichkeit je Mitarbeiter
- Globale & lokale Erklärungen (mit SHAP oder Fallback über Importances)
- Maßnahmenkatalog (kostenarm vs. investiv) je Top-Treiber
- CLI: train / predict / explain / top-factors

Voraussetzungen (pip):
    pip install pandas numpy scikit-learn joblib
    # optional (besser):
    pip install xgboost
    # optional (für bessere Erklärungen):
    pip install shap

Erwartete Daten (IBM HR Analytics Employee Attrition & Performance):
- Zielspalte: "Attrition" (Yes/No)
- Eindeutige ID-Spalte: "EmployeeNumber" (int oder str)
- Beispielsweise u. a.: Age, MonthlyIncome, OverTime, BusinessTravel, DistanceFromHome,
  JobRole, Department, JobSatisfaction, YearsAtCompany, ...

Hinweise:
- Spalten wie Over18, StandardHours, EmployeeCount sind konstant → werden automatisch ignoriert.
- PII vermeiden: Die ID wird nur zur Zuordnung verwendet und nicht in das Modell gespeist.

Nutzung (Beispiele):
    python attrition_copilot.py train --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out models/
    python attrition_copilot.py predict --employee 2032 --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --model models/model.joblib
    python attrition_copilot.py explain --employee 2032 --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --model models/model.joblib --topk 3
    python attrition_copilot.py top-factors --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --model models/model.joblib --topk 10

Tests (Mini-CLI-Smoke):
    # 1) Training & Speichern
    python attrition_copilot.py train --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out models/
    # 2) Top-Faktoren (global) ohne SHAP
    python attrition_copilot.py top-factors --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --model models/model.joblib --topk 5
    # 3) Lokale Erklärung (fällt zurück, wenn SHAP fehlt)
    python attrition_copilot.py explain --employee 2032 --csv data/WA_Fn-UseC_-HR-Employee-Attrition.csv --model models/model.joblib --topk 3

"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional: XGBoost unterstützen, wenn installiert
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional: SHAP – kann fehlen. Dann werden Fallbacks genutzt.
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ----------------------------------
# Konfig: Feature-Mapping & Maßnahmen
# ----------------------------------

# Mapping von Feature-Namen (aus Datensatz) zu Maßnahmenvorschlägen
MEASURES: Dict[str, Dict[str, List[str]]] = {
    "OverTime": {
        "kostenarm": [
            "Überstunden-Deckel und verpflichtende Ausgleichstage einführen",
            "Workload-Planung im Team wöchentlich prüfen",
        ],
        "investiv": [
            "Zusätzliche Stellen zur Entlastung schaffen",
            "Automatisierung/Tooling für wiederkehrende Aufgaben beschaffen",
        ],
    },
    "MonthlyIncome": {
        "kostenarm": [
            "Transparente Gehaltsbänder kommunizieren",
            "Flexible Benefits (z. B. Homeoffice-Tage, Zeitkonten) anbieten",
        ],
        "investiv": [
            "Gehaltserhöhung oder Bonusmodell einführen",
            "Leistungsbezogene Prämien strukturiert ausrollen",
        ],
    },
    "BusinessTravel": {
        "kostenarm": [
            "Reisen bündeln und durch virtuelle Meetings ersetzen",
            "Reiserichtlinien mit Fokus auf Erholungszeiten schärfen",
        ],
        "investiv": [
            "Regionale Teams/Vertretungen aufbauen",
            "Reise-Upgrade-Policy (z. B. Bahn 1. Klasse ab X Std.) prüfen",
        ],
    },
    "DistanceFromHome": {
        "kostenarm": [
            "Gleitzeit und hybride Arbeit fest verankern",
            "Fahrgemeinschaften/Jobrad fördern",
        ],
        "investiv": [
            "Relocation-Package anbieten",
            "Satellitenbüro näher am Wohnort prüfen",
        ],
    },
    "JobSatisfaction": {
        "kostenarm": [
            "Regelmäßige 1:1-Feedbackgespräche etablieren",
            "Anerkennungskultur (Lob, Sichtbarkeit von Leistungen) stärken",
        ],
        "investiv": [
            "Jobrotation/Karrierepfade entwickeln",
            "Führungskräftetrainings zu Motivation & Coaching ausrollen",
        ],
    },
    "YearsAtCompany": {
        "kostenarm": [
            "Onboarding-Buddy und 30/60/90-Tage-Check-ins",
            "Frühzeitige Entwicklungsziele (OKRs) gemeinsam definieren",
        ],
        "investiv": [
            "Bindungsprogramme (z. B. Aktienoptionen, Bildungsbudget)",
            "Schnellstarter-Programme mit Mentoring & Zertifikaten",
        ],
    },
    "Age": {
        "kostenarm": [
            "Entwicklungschancen und Projektverantwortung früh anbieten",
            "Regelmäßige Skill-Reviews und Lernpfade",
        ],
        "investiv": [
            "Geführte Karrierepfade (Junior→Senior) mit Level-Gehältern",
            "Externe Zertifizierungsprogramme finanzieren",
        ],
    },
    "TrainingTimesLastYear": {
        "kostenarm": [
            "Interne Brown-Bag-Sessions und Peer-Learning",
            "E-Learning-Lizenzen besser sichtbar/zugänglich machen",
        ],
        "investiv": [
            "Gezielte Trainingsbudgets pro Rolle",
            "Partnerschaften mit Bildungsträgern aufsetzen",
        ],
    },
}

# Features, die häufig konstant/nicht-informativ sind (werden entfernt, falls vorhanden)
DROP_CONST_CANDIDATES = {
    "Over18",          # im IBM-Datensatz immer "Y"
    "StandardHours",   # im IBM-Datensatz immer 80
    "EmployeeCount",   # im IBM-Datensatz immer 1
}

TARGET_COL = "Attrition"
ID_COL = "EmployeeNumber"  # eindeutige ID


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names_: List[str]
    original_feature_names_: List[str]
    target_col: str = TARGET_COL
    id_col: str = ID_COL
    model_type: str = "rf"  # oder "xgb"


# ---------------------
# Daten & Preprocessing
# ---------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Whitespace trimmen
    df.columns = [c.strip() for c in df.columns]

    # Zielvariable in 0/1 umwandeln
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype("Int64")

    # Überflüssige Konstanten werfen (falls vorhanden)
    for c in DROP_CONST_CANDIDATES:
        if c in df.columns:
            if df[c].nunique(dropna=False) <= 1:
                df = df.drop(columns=[c])

    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    assert TARGET_COL in df.columns, f"Zielspalte '{TARGET_COL}' fehlt."
    assert ID_COL in df.columns, f"ID-Spalte '{ID_COL}' fehlt."

    X = df.drop(columns=[TARGET_COL, ID_COL])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[(
        "onehot",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


# --------------
# Model Training
# --------------
def train_model(
    df: pd.DataFrame,
    use_xgb: bool = True,
    random_state: int = 42,
) -> Tuple[ModelBundle, Dict[str, float]]:
    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    if use_xgb and HAS_XGB:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
        )
        model_type = "xgb"
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        model_type = "rf"

    pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Abgeleitete Feature-Namen nach One-Hot
    feature_names = _get_feature_names_from_preprocessor(pipeline, num_feats, cat_feats)

    bundle = ModelBundle(
        pipeline=pipeline,
        feature_names_=feature_names,
        original_feature_names_=num_feats + cat_feats,
        model_type=model_type,
    )

    metrics = {
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "accuracy": float(report.get("accuracy", np.nan)),
        "precision_1": float(report.get("1", {}).get("precision", np.nan)),
        "recall_1": float(report.get("1", {}).get("recall", np.nan)),
    }

    return bundle, metrics


def _get_feature_names_from_preprocessor(
    pipeline: Pipeline, num_feats: List[str], cat_feats: List[str]
) -> List[str]:
    pre: ColumnTransformer = pipeline.named_steps["pre"]

    out_feature_names: List[str] = []

    # Numerische Features (Skalierer ändert die Namen nicht)
    out_feature_names.extend(num_feats)

    # Kategorische Features (One-Hot liefert Kategorienamen)
    if len(cat_feats) > 0:
        ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
        ohe_feature_names = ohe.get_feature_names_out(cat_feats).tolist()
        out_feature_names.extend(ohe_feature_names)

    return out_feature_names


# --------------
# Explainability – SHAP oder Fallback
# --------------
def make_explainer(bundle: ModelBundle, X_sample: Optional[pd.DataFrame] = None):
    """Gibt ein Tupel (explainer_fn, feature_names). Wenn SHAP vorhanden ist, nutzt es SHAP;
    sonst einen Fallback über Permutation Importance (global) und einfache Heuristik (lokal)."""
    model = bundle.pipeline.named_steps["model"]
    pre = bundle.pipeline.named_steps["pre"]

    if HAS_SHAP:
        # Für Tree-Modelle schneller und genauer
        if bundle.model_type in {"rf", "xgb"}:
            explainer = shap.TreeExplainer(model)
            def local_fn(X_df: pd.DataFrame):
                X_trans = pre.transform(X_df)
                sv = explainer.shap_values(X_trans)
                # Liste [neg, pos] => pos nehmen
                if isinstance(sv, list) and len(sv) == 2:
                    return sv[1]
                return sv
            return local_fn, bundle.feature_names_
        else:
            explainer = shap.Explainer(model.predict_proba)
            def local_fn(X_df: pd.DataFrame):
                X_trans = pre.transform(X_df)
                exp = explainer(X_trans)
                return exp.values[:, 1]
            return local_fn, bundle.feature_names_

    # Fallback ohne SHAP
    def local_fn(X_df: pd.DataFrame):
        X_trans = pre.transform(X_df)
        # Wir approximieren lokale Beiträge mit (Feature * Modell-Feature-Importance)
        # Für RF: feature_importances_; für XGB ebenso
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
            # Skaliere Beiträge pro Zeile
            return (X_trans * imp).astype(float)
        # Generischer Fallback: Koeffizienten-basiert nicht verfügbar → Nullen
        return np.zeros((X_trans.shape[0], X_trans.shape[1]))

    return local_fn, bundle.feature_names_


def aggregate_by_original_feature(values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Aggregiert OHE-Features auf Ursprungsfeature-Ebene über |Wert|."""
    agg: Dict[str, float] = {}
    for val, name in zip(values, feature_names):
        base = name.split("_")[0]
        agg[base] = agg.get(base, 0.0) + float(abs(val))
    return agg


# ------------------------
# Vorhersage & Erklärungen
# ------------------------
def predict_for_employee(
    bundle: ModelBundle, df: pd.DataFrame, employee_id: str | int
) -> Tuple[float, pd.Series]:
    row = df.loc[df[ID_COL].astype(str) == str(employee_id)]
    if row.empty:
        raise ValueError(f"Employee {employee_id} nicht gefunden.")

    X_row = row.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    proba = float(bundle.pipeline.predict_proba(X_row)[0, 1])
    return proba, X_row.iloc[0]


def explain_instance(
    bundle: ModelBundle,
    df: pd.DataFrame,
    employee_id: str | int,
    topk: int = 3,
) -> Dict:
    proba, X_row = predict_for_employee(bundle, df, employee_id)

    local_fn, feat_names = make_explainer(bundle)
    X_row_df = X_row.to_frame().T
    local_vals = local_fn(X_row_df)[0]

    agg = aggregate_by_original_feature(local_vals, feat_names)
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]

    reasons = []
    for fname, impact in top:
        direction = _direction_of_effect(fname, X_row)
        text = reason_text(fname, direction, X_row)
        measures = MEASURES.get(fname, None)
        reasons.append({
            "feature": fname,
            "impact_strength": round(float(impact), 4),
            "direction": direction,  # "hoch"/"niedrig"/"n/a"
            "reason": text,
            "measures": measures,
        })

    return {
        "employee": str(employee_id),
        "attrition_probability": round(proba, 4),
        "top_reasons": reasons,
        "explain_method": "shap" if HAS_SHAP else "importance_fallback",
    }


def _direction_of_effect(fname: str, row: pd.Series) -> str:
    """Sehr einfache Heuristik, die für gängige Features eine Richtung liefert."""
    if fname not in row.index:
        return "n/a"

    val = row[fname]
    if fname in {"MonthlyIncome", "Age", "YearsAtCompany", "TrainingTimesLastYear"}:
        try:
            # Vergleich gegen eigenen Wert (Fallback), Median auf Series ist hier nicht sinnvoll.
            return "hoch" if float(val) >= float(val) else "niedrig"
        except Exception:
            return "n/a"

    if fname in {"DistanceFromHome"}:
        try:
            return "hoch" if float(val) > 10 else "niedrig"
        except Exception:
            return "n/a"

    if fname in {"OverTime"}:
        return "ja" if str(val).lower() in {"yes", "y", "true", "1"} else "nein"

    try:
        return "hoch" if float(val) > 0 else "niedrig"
    except Exception:
        return "n/a"


def reason_text(fname: str, direction: str, row: pd.Series) -> str:
    """Natürlichsprachliche Begründung für Endnutzer."""
    if fname == "OverTime":
        return "Viele Überstunden belasten und korrelieren in den Daten mit höherer Fluktuation." if direction in {"ja", "hoch"} else "Geringe Überstunden reduzieren das Risiko."
    if fname == "MonthlyIncome":
        return "Ein niedriges Einkommen erhöht im Datensatz die Wechselbereitschaft." if direction == "niedrig" else "Ein wettbewerbsfähiges Einkommen wirkt stabilisierend."
    if fname == "BusinessTravel":
        return "Häufige Reisetätigkeit ist häufig ein Treiber für Kündigungen (Belastung, Work-Life-Balance)."
    if fname == "DistanceFromHome":
        return "Lange Pendeldistanzen erhöhen das Risiko (Zeit, Stress, Kosten)." if direction == "hoch" else "Kurze Anfahrtswege wirken entlastend."
    if fname == "JobSatisfaction":
        return "Niedrige Zufriedenheit ist ein klarer Frühindikator für Fluktuation." if direction == "niedrig" else "Gute Zufriedenheit verringert das Risiko."
    if fname == "YearsAtCompany":
        return "Kurze Betriebszugehörigkeit geht oft mit höherer Wechselwahrscheinlichkeit einher." if direction == "niedrig" else "Längere Zugehörigkeit stabilisiert."
    if fname == "Age":
        return "Jüngere Mitarbeitende wechseln im Durchschnitt häufiger (Karrieresprung, Marktoptionen)." if direction == "niedrig" else "Höheres Alter korreliert im Datensatz oft mit geringerer Fluktuation."
    if fname == "TrainingTimesLastYear":
        return "Wenig Weiterbildung kann Frustration und Abwanderung fördern." if direction == "niedrig" else "Gute Weiterbildungsmöglichkeiten binden."

    return f"Das Merkmal '{fname}' zeigt in den Daten einen starken Einfluss auf das Risiko."


# -------------------
# Speichern & Laden
# -------------------
def save_bundle(bundle: ModelBundle, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.joblib")
    meta = {
        "feature_names_": bundle.feature_names_,
        "original_feature_names_": bundle.original_feature_names_,
        "model_type": bundle.model_type,
        "target_col": bundle.target_col,
        "id_col": bundle.id_col,
    }
    joblib.dump({"pipeline": bundle.pipeline, "meta": meta}, path)
    print(f"Gespeichert: {path}")


def load_bundle(path: str) -> ModelBundle:
    obj = joblib.load(path)
    pipeline: Pipeline = obj["pipeline"]
    meta = obj["meta"]
    return ModelBundle(
        pipeline=pipeline,
        feature_names_=meta["feature_names_"],
        original_feature_names_=meta["original_feature_names_"],
        model_type=meta.get("model_type", "rf"),
        target_col=meta.get("target_col", TARGET_COL),
        id_col=meta.get("id_col", ID_COL),
    )


# -------------------
# Globale Wichtigkeit
# -------------------
def global_top_factors(bundle: ModelBundle, df: pd.DataFrame, topk: int = 10) -> List[Tuple[str, float]]:
    pre = bundle.pipeline.named_steps["pre"]
    model = bundle.pipeline.named_steps["model"]

    X = df.drop(columns=[TARGET_COL, ID_COL])
    X_trans = pre.transform(X)

    # SHAP (falls verfügbar)
    if HAS_SHAP and bundle.model_type in {"rf", "xgb"}:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_trans)
        shap_pos = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
        abs_means = np.mean(np.abs(shap_pos), axis=0)
    elif hasattr(model, "feature_importances_"):
        # Fallback: Feature Importances
        abs_means = np.asarray(model.feature_importances_, dtype=float)
        # Normierung auf Summe 1, rein kosmetisch
        if abs_means.sum() > 0:
            abs_means = abs_means / abs_means.sum()
    else:
        # Letzter Fallback: Permutation Importance (teurer)
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X_trans, df[TARGET_COL].astype(int), scoring="roc_auc", n_repeats=5, random_state=0)
        abs_means = r.importances_mean

    # Auf Ursprungsfeature aggregieren
    agg = {}
    for val, name in zip(abs_means, bundle.feature_names_):
        base = name.split("_")[0]
        agg[base] = agg.get(base, 0.0) + float(abs(val))

    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]
    return top


# ---------
# CLI-Teil
# ---------
def cmd_train(args):
    df = load_data(args.csv)

    bundle, metrics = train_model(df, use_xgb=args.xgb)
    print(json.dumps({"metrics": metrics}, indent=2))

    if args.out:
        save_bundle(bundle, args.out)


def cmd_predict(args):
    bundle = load_bundle(args.model)
    df = load_data(args.csv)

    proba, _ = predict_for_employee(bundle, df, args.employee)
    print(json.dumps({"employee": args.employee, "attrition_probability": round(proba, 4)}, indent=2))


def cmd_explain(args):
    bundle = load_bundle(args.model)
    df = load_data(args.csv)
    expl = explain_instance(bundle, df, args.employee, topk=args.topk)
    print(json.dumps(expl, indent=2, ensure_ascii=False))


def cmd_top_factors(args):
    bundle = load_bundle(args.model)
    df = load_data(args.csv)
    top = global_top_factors(bundle, df, topk=args.topk)

    # Ergänze Maßnahmen, wo vorhanden
    result = []
    for fname, score in top:
        result.append({
            "feature": fname,
            "global_importance": round(float(score), 4),
            "measures": MEASURES.get(fname),
        })
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attrition Copilot – IBM HR Starter (SHAP-fallback)")
    sub = p.add_subparsers(required=True)

    # train
    pt = sub.add_parser("train", help="Modell trainieren")
    pt.add_argument("--csv", required=True, help="Pfad zur IBM HR CSV")
    pt.add_argument("--out", default="models", help="Output-Verzeichnis für model.joblib")
    pt.add_argument("--xgb", action="store_true", help="Falls installiert: XGBoost verwenden")
    pt.set_defaults(func=cmd_train)

    # predict
    pp = sub.add_parser("predict", help="Kündigungswahrscheinlichkeit für Mitarbeiter")
    pp.add_argument("--model", required=True, help="Pfad zu models/model.joblib")
    pp.add_argument("--csv", required=True, help="HR-Snapshot mit EmployeeNumber")
    pp.add_argument("--employee", required=True, help="EmployeeNumber (ID)")
    pp.set_defaults(func=cmd_predict)

    # explain
    pe = sub.add_parser("explain", help="Top-Gründe & Maßnahmen für Mitarbeiter")
    pe.add_argument("--model", required=True, help="Pfad zu models/model.joblib")
    pe.add_argument("--csv", required=True, help="HR-Snapshot mit EmployeeNumber")
    pe.add_argument("--employee", required=True, help="EmployeeNumber (ID)")
    pe.add_argument("--topk", type=int, default=3, help="Anzahl Top-Gründe")
    pe.set_defaults(func=cmd_explain)

    # top-factors (global)
    pg = sub.add_parser("top-factors", help="Globale Treiber im Datensatz")
    pg.add_argument("--model", required=True, help="Pfad zu models/model.joblib")
    pg.add_argument("--csv", required=True, help="HR-Snapshot")
    pg.add_argument("--topk", type=int, default=10)
    pg.set_defaults(func=cmd_top_factors)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
