# -*- coding: utf-8 -*-
"""
02_generar_kpis_inicio.py
Genera KPIs iniciales para el dashboard HIS11.

Entrada requerida:
- data/outputs/episodios_dashboard.csv

Salidas:
- kpis_inicio.csv
- kpis_inicio_estatus.csv
- kpis_inicio_retraso.csv
- kpis_inicio_rangos.csv
"""

from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/outputs/episodios_dashboard.csv")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Generando KPIs de inicio...")
    ep = pd.read_csv(INPUT_PATH, low_memory=False)

    t = pd.to_numeric(ep["tiempo_alta_admin_horas"], errors="coerce")
    total = len(ep)
    con_am = ep["tiene_alta_medica"].sum()
    con_aa = ep["tiene_alta_administrativa"].sum()
    pendientes = (ep["estatus_cierre"] == "Pendiente").sum()
    cerrados_ok = (ep["estatus_cierre"] == "Cerrado con tiempo calculable").sum()
    cerrados_incon = (ep["estatus_cierre"] == "Cerrado con datos inconsistentes").sum()
    cerrados_sin_am = (ep["estatus_cierre"] == "Cerrado sin alta médica").sum()
    sin_trazabilidad = (ep["estatus_cierre"] == "Sin trazabilidad").sum()
    calculables = t.notna().sum()
    criticos_total = (ep["nivel_retraso"] == "Crítico").sum()
    criticos_calc = (ep[t.notna()]["nivel_retraso"] == "Crítico").sum()

    tiempo_vals = t.dropna()
    prom_tiempo = round(tiempo_vals.mean(), 2) if len(tiempo_vals) else 0
    mediana_tiempo = round(tiempo_vals.median(), 2) if len(tiempo_vals) else 0
    pct_mas_24h = round(100 * (tiempo_vals > 24).sum() / max(len(tiempo_vals), 1), 2)
    pct_crit_total = round(100 * criticos_total / max(total, 1), 2)
    pct_crit_calc = round(100 * criticos_calc / max(calculables, 1), 2)

    kpis_inicio = pd.DataFrame([
        {"kpi": "Total episodios analizados", "valor": total, "formato": "numero",
         "descripcion": "Número total de episodios únicos (folio+ext+área) en la base analizada."},
        {"kpi": "Episodios con alta médica", "valor": int(con_am), "formato": "numero",
         "descripcion": "Episodios que tienen fecha y hora de alta médica registrada en Hospac."},
        {"kpi": "Episodios con alta administrativa", "valor": int(con_aa), "formato": "numero",
         "descripcion": "Episodios que tienen fecha y hora de alta administrativa registrada en Hospac."},
        {"kpi": "Episodios pendientes (alta médica sin cierre admin)", "valor": int(pendientes), "formato": "numero",
         "descripcion": "Tienen alta médica pero aún no tienen alta administrativa — son los casos activos del Monitor Operativo."},
        {"kpi": "Cerrados con tiempo calculable", "valor": int(cerrados_ok), "formato": "numero",
         "descripcion": "Alta admin registrada después del alta médica; tiempo_alta_admin_horas está disponible para análisis."},
        {"kpi": "Cerrados con datos inconsistentes (admin antes que médica)", "valor": int(cerrados_incon), "formato": "numero",
         "descripcion": "Alta administrativa registrada antes que la médica en la fuente; el tiempo resultaría negativo y se excluye."},
        {"kpi": "Cerrados sin alta médica registrada", "valor": int(cerrados_sin_am), "formato": "numero",
         "descripcion": "Tienen alta administrativa pero no alta médica; no es posible calcular el tiempo del proceso."},
        {"kpi": "Sin trazabilidad suficiente", "valor": int(sin_trazabilidad), "formato": "numero",
         "descripcion": "Episodios sin alta médica ni administrativa registrada; existen en Hosfol pero sin datos de cierre."},
        {"kpi": "Tiempo promedio alta admin (horas)", "valor": prom_tiempo, "formato": "horas",
         "descripcion": "Promedio de horas entre alta médica y alta administrativa, calculado solo sobre episodios con ambas fechas válidas."},
        {"kpi": "Mediana tiempo alta admin (horas)", "valor": mediana_tiempo, "formato": "horas",
         "descripcion": "Mediana del mismo tiempo; menos sensible a valores extremos que el promedio."},
        {"kpi": "% casos con retraso > 24h", "valor": pct_mas_24h, "formato": "porcentaje",
         "descripcion": "Porcentaje de episodios con tiempo_alta_admin_horas mayor a 24 horas."},
        {"kpi": "% casos críticos sobre total (pct_casos_criticos_total)", "valor": pct_crit_total, "formato": "porcentaje",
         "descripcion": f"Porcentaje de episodios con nivel_retraso=Crítico (≥72h) sobre los {total:,} episodios totales del dashboard."},
        {"kpi": "% casos críticos sobre calculables (pct_criticos_casos_calculables)", "valor": pct_crit_calc, "formato": "porcentaje",
         "descripcion": f"Porcentaje de episodios con nivel_retraso=Crítico (≥72h) sobre los {calculables:,} episodios con tiempo calculable."},
    ])

    kpis_inicio = kpis_inicio[["kpi", "valor", "descripcion", "formato"]]
    kpis_inicio.to_csv(OUTPUT_DIR / "kpis_inicio.csv", index=False)

    ep.groupby("estatus_cierre").size().reset_index(name="conteo_episodios") \
        .to_csv(OUTPUT_DIR / "kpis_inicio_estatus.csv", index=False)

    ep.groupby("nivel_retraso").size().reset_index(name="conteo_episodios") \
        .to_csv(OUTPUT_DIR / "kpis_inicio_retraso.csv", index=False)

    bins = [0, 8, 24, 48, 72, float("inf")]
    labs = ["0–8h", "8–24h", "24–48h", "48–72h", ">72h"]
    ep["rango_tiempo"] = pd.cut(t, bins=bins, labels=labs, right=True)
    ep.groupby("rango_tiempo", observed=True).size().reset_index(name="conteo_episodios") \
        .to_csv(OUTPUT_DIR / "kpis_inicio_rangos.csv", index=False)

    print("Listo: kpis_inicio.csv y auxiliares de KPIs")


if __name__ == "__main__":
    main()
