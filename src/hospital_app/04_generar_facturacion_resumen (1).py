# -*- coding: utf-8 -*-
"""
04_generar_facturacion_resumen.py
Genera facturacion_resumen.csv para el dashboard HIS11.

Entrada requerida:
- data/outputs/episodios_dashboard.csv

Salida:
- facturacion_resumen.csv
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


def main():
    print("Generando facturacion_resumen.csv...")
    ep = pd.read_csv(INPUT_PATH, low_memory=False)
    ep["tiempo_alta_admin_horas"] = pd.to_numeric(ep["tiempo_alta_admin_horas"], errors="coerce")
    ep["tiempo_ciclo_facturacion_horas"] = pd.to_numeric(ep["tiempo_ciclo_facturacion_horas"], errors="coerce")
    ep["num_facturas"] = pd.to_numeric(ep["num_facturas"], errors="coerce").fillna(0).astype(int)

    ep_fac = ep[ep["num_facturas"] > 0].copy()
    prom_fac = round(ep_fac["num_facturas"].mean(), 2) if not ep_fac.empty else 0
    pct_multi = round(100 * (ep_fac["num_facturas"] > 1).sum() / max(len(ep_fac), 1), 2)
    prom_ciclo = round(ep_fac["tiempo_ciclo_facturacion_horas"].dropna().mean(), 2) if not ep_fac.empty else 0
    max_fac_ep = ep_fac.loc[ep_fac["num_facturas"].idxmax(), "episodio_id"] if not ep_fac.empty else ""
    max_fac_n = int(ep_fac["num_facturas"].max()) if not ep_fac.empty else 0

    kpis_fac = pd.DataFrame([
        {"kpi": "Promedio facturas por episodio", "valor": prom_fac, "formato": "numero",
         "descripcion": "Promedio de facturas por episodio considerando solo los que tienen al menos una factura.", "tipo_fila": "kpi_global"},
        {"kpi": "% episodios con más de 1 factura", "valor": pct_multi, "formato": "porcentaje",
         "descripcion": "Porcentaje de episodios con facturación que tienen más de una factura emitida.", "tipo_fila": "kpi_global"},
        {"kpi": "Tiempo promedio ciclo facturación (h)", "valor": prom_ciclo, "formato": "horas",
         "descripcion": "Promedio de horas entre la primera y la última factura (solo episodios con más de una factura).", "tipo_fila": "kpi_global"},
        {"kpi": "Episodio con más facturas", "valor": max_fac_ep, "formato": "texto",
         "descripcion": "ID del episodio que concentra el mayor número de facturas en toda la base.", "tipo_fila": "kpi_global"},
        {"kpi": "Número máximo de facturas", "valor": max_fac_n, "formato": "numero",
         "descripcion": "Cantidad máxima de facturas registradas en un solo episodio.", "tipo_fila": "kpi_global"},
    ])

    nf = ep.groupby("num_facturas").agg(
        conteo_episodios=("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas=("tiempo_alta_admin_horas", "mean"),
        tiempo_mediana_alta_admin_horas=("tiempo_alta_admin_horas", "median"),
    ).reset_index().round(2)
    nf["tipo_fila"] = "por_num_facturas"

    grupo_map = {"Sin factura": "0 facturas", "1 factura": "1 factura", "Múltiples facturas": "2+ facturas"}
    ep["grupo_label"] = ep["grupo_facturacion"].map(grupo_map).fillna(ep["grupo_facturacion"])
    grp = ep.groupby("grupo_label").agg(
        conteo_episodios=("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas=("tiempo_alta_admin_horas", "mean"),
        tiempo_mediana_alta_admin_horas=("tiempo_alta_admin_horas", "median"),
    ).reset_index().rename(columns={"grupo_label": "grupo_facturacion"}).round(2)
    orden_g = {"0 facturas": 0, "1 factura": 1, "2+ facturas": 2}
    grp = grp.assign(_o=grp["grupo_facturacion"].map(orden_g)).sort_values("_o").drop(columns="_o").reset_index(drop=True)
    grp["tipo_fila"] = "por_grupo_facturacion"

    top_cols_f = ["episodio_id", "folio", "ext", "num_facturas",
                  "fecha_primera_factura", "fecha_ultima_factura",
                  "tiempo_ciclo_facturacion_horas", "tiempo_alta_admin_horas",
                  "nivel_retraso", "estatus_cierre"]

    top20_f = ep[ep["num_facturas"] > 0].sort_values("num_facturas", ascending=False).head(20)[top_cols_f].reset_index(drop=True)
    top20_f["motivo_sin_tiempo"] = top20_f.apply(motivo_sin_tiempo, axis=1)
    top20_f["tipo_fila"] = "top_episodios"

    top_calc_f = ep[(ep["num_facturas"] > 0) & ep["tiempo_alta_admin_horas"].notna()] \
        .sort_values("num_facturas", ascending=False).head(20)[top_cols_f].reset_index(drop=True)
    top_calc_f["motivo_sin_tiempo"] = ""
    top_calc_f["tipo_fila"] = "top_episodios_con_tiempo_calculable"

    all_cols_f = ["tipo_fila", "kpi", "valor", "descripcion", "formato",
                  "num_facturas", "grupo_facturacion",
                  "conteo_episodios", "tiempo_promedio_alta_admin_horas", "tiempo_mediana_alta_admin_horas",
                  "episodio_id", "folio", "ext", "fecha_primera_factura", "fecha_ultima_factura",
                  "tiempo_ciclo_facturacion_horas", "tiempo_alta_admin_horas",
                  "nivel_retraso", "motivo_sin_tiempo", "estatus_cierre"]

    fac_res = pd.concat([
        align(kpis_fac, all_cols_f),
        align(nf, all_cols_f),
        align(grp, all_cols_f),
        align(top20_f, all_cols_f),
        align(top_calc_f, all_cols_f),
    ], ignore_index=True)

    fac_res.to_csv(OUTPUT_DIR / "facturacion_resumen.csv", index=False)
    print("Listo: facturacion_resumen.csv")


if __name__ == "__main__":
    main()
