# -*- coding: utf-8 -*-
"""
07_generar_monitor_operativo.py
Genera monitor_operativo.csv para el dashboard HIS11.

Entrada requerida:
- data/outputs/episodios_dashboard.csv

Salida:
- monitor_operativo.csv
"""

from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/outputs/episodios_dashboard.csv")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Generando monitor_operativo.csv...")
    ep = pd.read_csv(INPUT_PATH, low_memory=False)

    ep["fecha_alta_medica_dt"] = pd.to_datetime(ep["fecha_alta_medica"].replace("", pd.NA), errors="coerce")
    fecha_referencia = ep["fecha_alta_medica_dt"].max()

    ep["horas_desde_alta_medica"] = pd.to_numeric(ep["horas_desde_alta_medica"], errors="coerce")

    # Normalizar booleanos por si pandas lee True/False como texto.
    for col in ["tiene_alta_medica", "tiene_alta_administrativa", "tiene_facturas", "tiene_transacciones", "tiene_requisiciones"]:
        if ep[col].dtype == object:
            ep[col] = ep[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

    monitor = ep[
        ep["tiene_alta_medica"].fillna(False)
        & (~ep["tiene_alta_administrativa"].fillna(False))
        & ep["fecha_alta_medica_dt"].notna()
    ].copy()

    monitor = monitor.sort_values("horas_desde_alta_medica", ascending=False, na_position="last").reset_index(drop=True)
    monitor["fecha_referencia"] = fecha_referencia.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(fecha_referencia) else ""

    mon_cols = [
        "episodio_id", "paciente_id", "folio", "ext", "area_principal",
        "fecha_alta_medica",
        "horas_desde_alta_medica", "fecha_referencia",
        "nivel_prioridad",
        "num_facturas", "num_transacciones", "num_requisiciones",
        "dep_sol_principal", "tipo_habitacion", "estatus_cierre",
        "tiene_facturas", "tiene_transacciones", "tiene_requisiciones",
    ]

    monitor[mon_cols].to_csv(OUTPUT_DIR / "monitor_operativo.csv", index=False)
    print(f"Listo: monitor_operativo.csv -> {len(monitor):,} episodios pendientes")
    print(f"fecha_referencia: {monitor['fecha_referencia'].iloc[0] if len(monitor) else ''}")


if __name__ == "__main__":
    main()
