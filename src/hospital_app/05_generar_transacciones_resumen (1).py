# -*- coding: utf-8 -*-
"""
05_generar_transacciones_resumen.py
Genera transacciones_resumen.csv para el dashboard HIS11.

Entrada requerida:
- data/outputs/episodios_dashboard.csv

Salida:
- transacciones_resumen.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/outputs/episodios_dashboard.csv")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def motivo_sin_tiempo(row):
    if pd.notna(row.get("tiempo_alta_admin_horas")) and not pd.isna(row.get("tiempo_alta_admin_horas")):
        return ""
    return {
        "Sin trazabilidad": "Sin alta médica ni administrativa",
        "Cerrado sin alta médica": "Alta administrativa sin alta médica",
        "Cerrado con datos inconsistentes": "Alta admin registrada antes que médica",
        "Pendiente": "Pendiente de cierre administrativo",
    }.get(str(row.get("estatus_cierre", "")), "Sin datos suficientes")


def align(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].copy()


def grupo_trans(n):
    if n == 0:
        return "0 transacciones"
    if n <= 5:
        return "1–5 transacciones"
    if n <= 10:
        return "6–10 transacciones"
    if n <= 20:
        return "11–20 transacciones"
    return ">20 transacciones"


def main():
    print("Generando transacciones_resumen.csv...")
    ep = pd.read_csv(INPUT_PATH, low_memory=False)
    total = len(ep)

    for col in ["num_transacciones", "num_transacciones_negativas", "num_transacciones_canceladas"]:
        ep[col] = pd.to_numeric(ep[col], errors="coerce").fillna(0).astype(int)
    for col in ["monto_total_transacciones", "tiempo_alta_admin_horas"]:
        ep[col] = pd.to_numeric(ep[col], errors="coerce")

    kpis_trans = pd.DataFrame([
        {"kpi": "Promedio transacciones por episodio",
         "valor": round(ep["num_transacciones"].mean(), 2), "formato": "numero",
         "descripcion": f"Promedio de movimientos por episodio considerando los {total:,} episodios totales (incluye episodios con 0).",
         "tipo_fila": "kpi_global"},
        {"kpi": "% episodios con transacciones negativas",
         "valor": round(100 * (ep["num_transacciones_negativas"] > 0).sum() / max(total, 1), 2), "formato": "porcentaje",
         "descripcion": "Porcentaje de episodios con al menos una transacción negativa. Las transacciones negativas son parte normal del flujo administrativo.",
         "tipo_fila": "kpi_global"},
        {"kpi": "% episodios con cancelaciones",
         "valor": round(100 * (ep["num_transacciones_canceladas"] > 0).sum() / max(total, 1), 2), "formato": "porcentaje",
         "descripcion": "Porcentaje de episodios con al menos una transacción cancelada.",
         "tipo_fila": "kpi_global"},
        {"kpi": "Monto total transacciones",
         "valor": round(ep["monto_total_transacciones"].sum(), 2), "formato": "dinero",
         "descripcion": "Suma del monto total de todas las transacciones en el periodo analizado.",
         "tipo_fila": "kpi_global"},
    ])

    orden_trans = {"0 transacciones": 0, "1–5 transacciones": 1, "6–10 transacciones": 2,
                   "11–20 transacciones": 3, ">20 transacciones": 4}
    ep["grupo_trans"] = ep["num_transacciones"].apply(grupo_trans)
    grp_t = ep.groupby("grupo_trans").agg(
        conteo_episodios=("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas=("tiempo_alta_admin_horas", "mean"),
        tiempo_mediana_alta_admin_horas=("tiempo_alta_admin_horas", "median"),
    ).reset_index().rename(columns={"grupo_trans": "grupo_transacciones"}).round(2)
    grp_t = grp_t.assign(_o=grp_t["grupo_transacciones"].map(orden_trans)).sort_values("_o").drop(columns="_o").reset_index(drop=True)
    grp_t["tipo_fila"] = "por_grupo_transacciones"

    ep["tiene_neg"] = ep["num_transacciones_negativas"] > 0
    neg_comp = ep.groupby("tiene_neg").agg(
        conteo_episodios=("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas=("tiempo_alta_admin_horas", "mean"),
        tiempo_mediana_alta_admin_horas=("tiempo_alta_admin_horas", "median"),
    ).reset_index().round(2)
    neg_comp["tiene_transacciones_negativas"] = neg_comp["tiene_neg"].map({
        True: "Con transacciones negativas",
        False: "Sin transacciones negativas",
    })
    neg_comp = neg_comp.drop(columns="tiene_neg")
    neg_comp["tipo_fila"] = "comparativo_negativas"

    top_cols_t = ["episodio_id", "folio", "ext", "num_transacciones",
                  "num_transacciones_negativas", "num_transacciones_canceladas",
                  "monto_total_transacciones", "tiempo_alta_admin_horas",
                  "nivel_retraso", "estatus_cierre"]

    top20_t = ep[ep["num_transacciones"] > 0].sort_values("num_transacciones", ascending=False).head(20)[top_cols_t].reset_index(drop=True)
    top20_t["motivo_sin_tiempo"] = top20_t.apply(motivo_sin_tiempo, axis=1)
    top20_t["tipo_fila"] = "top_episodios"

    top_calc_t = ep[(ep["num_transacciones"] > 0) & ep["tiempo_alta_admin_horas"].notna()] \
        .sort_values("num_transacciones", ascending=False).head(20)[top_cols_t].reset_index(drop=True)
    top_calc_t["motivo_sin_tiempo"] = ""
    top_calc_t["tipo_fila"] = "top_episodios_con_tiempo_calculable"

    all_cols_t = ["tipo_fila", "kpi", "valor", "descripcion", "formato",
                  "grupo_transacciones", "tiene_transacciones_negativas",
                  "conteo_episodios", "tiempo_promedio_alta_admin_horas", "tiempo_mediana_alta_admin_horas",
                  "episodio_id", "folio", "ext",
                  "num_transacciones", "num_transacciones_negativas", "num_transacciones_canceladas",
                  "monto_total_transacciones", "tiempo_alta_admin_horas",
                  "nivel_retraso", "motivo_sin_tiempo", "estatus_cierre"]

    trans_res = pd.concat([
        align(kpis_trans, all_cols_t),
        align(grp_t, all_cols_t),
        align(neg_comp, all_cols_t),
        align(top20_t, all_cols_t),
        align(top_calc_t, all_cols_t),
    ], ignore_index=True)

    trans_res.to_csv(OUTPUT_DIR / "transacciones_resumen.csv", index=False)
    print("Listo: transacciones_resumen.csv")


if __name__ == "__main__":
    main()
