# -*- coding: utf-8 -*-
"""
01_generar_episodios_dashboard.py
Genera episodios_dashboard.csv para el dashboard HIS11.

Fuentes requeridas en INPUT_DIR:
- Hospac.csv
- Hosfol.csv
- Hosffa.csv
- Hostransacciones.csv
- Hosder.csv

Salidas:
- episodios_dashboard.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Nombres de repo
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_fecha(serie):
    s = serie.astype(str).str.strip()
    return pd.to_datetime(s.where(s.str.len() == 8, other=pd.NA), format="%Y%m%d", errors="coerce")


def parse_hora(serie):
    s = serie.astype(str).str.strip().str.zfill(6)
    valida = s.str.match(r"^\d{6}$")
    hh = s.str[:2].where(valida).astype(float)
    mm = s.str[2:4].where(valida).astype(float)
    ss = s.str[4:6].where(valida).astype(float)
    return pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s", errors="coerce")


def fecha_hora_dt(fecha_serie, hora_serie):
    return parse_fecha(fecha_serie) + parse_hora(hora_serie)


def horas_entre(dt_inicio, dt_fin):
    diff = (dt_fin - dt_inicio).dt.total_seconds() / 3600
    return diff.where(diff >= 0)


def make_episodio_id(area, folio, ext):
    a = area.astype(str).str.strip()
    f = pd.to_numeric(folio, errors="coerce").astype("Int64").astype(str)
    e = pd.to_numeric(ext, errors="coerce").astype("Int64").astype(str)
    return a + "_" + f + "_" + e


def clasificar_retraso(h):
    if pd.isna(h):
        return "Sin dato"
    if h < 8:
        return "Sin retraso"
    if h < 24:
        return "Leve"
    if h < 48:
        return "Moderado"
    if h < 72:
        return "Alto"
    return "Crítico"


def clasificar_prioridad(h):
    if pd.isna(h):
        return "Sin dato"
    if h <= 24:
        return "Baja"
    if h <= 48:
        return "Media"
    if h <= 72:
        return "Alta"
    return "Crítica"


def main():
    print("Generando episodios_dashboard.csv...")

    pac = pd.read_csv(INPUT_DIR / "Hospac.csv", low_memory=False)
    fol = pd.read_csv(INPUT_DIR / "Hosfol.csv", low_memory=False)
    ffa = pd.read_csv(INPUT_DIR / "Hosffa.csv", low_memory=False)
    trans = pd.read_csv(INPUT_DIR / "Hostransacciones.csv", low_memory=False)
    der = pd.read_csv(INPUT_DIR / "Hosder.csv", low_memory=False)

    # 1. Episodios base: Hosfol
    fol = fol.copy()
    fol["folio"] = pd.to_numeric(fol["f_folio"], errors="coerce")
    fol["ext"] = pd.to_numeric(fol["f_folio_ext"], errors="coerce")
    fol["area"] = fol["f_area"].astype(str).str.strip()
    fol["paciente_id"] = fol["f_num_exp"].astype(str).str.strip()
    fol["dt_apertura"] = parse_fecha(fol["f_fec_ape"])
    fol["dt_cierre"] = parse_fecha(fol["f_fec_cie"])
    fol["episodio_id"] = make_episodio_id(fol["area"], fol["folio"], fol["ext"])

    fol_ep = (
        fol.sort_values("dt_apertura")
        .groupby("episodio_id", as_index=False)
        .last()
    )

    # 2. Altas: Hospac
    pac = pac.copy()
    pac["paciente_id_pac"] = pac["p_num_exp"].astype(str).str.strip()
    pac["episodio_id"] = make_episodio_id(pac["p_area"], pac["p_fol_cto"], pac["p_fol_ext"])
    pac["dt_alta_medica"] = fecha_hora_dt(pac["FechaAltaMedica"], pac["HoraAltaMedica"])
    pac["dt_alta_admin"] = fecha_hora_dt(pac["FechaAltaAdministrativa"], pac["HoraAltaAdministrativa"])
    pac["dt_ingreso"] = fecha_hora_dt(pac["p_fec_lld"], pac["p_hra_lld"])
    pac["dt_salida"] = fecha_hora_dt(pac["p_fec_sda"], pac["p_hra_sda"])
    pac["tipo_habitacion"] = pac["p_tpo_paq"].astype(str).str.strip()

    pac_ep = (
        pac[["episodio_id", "paciente_id_pac", "dt_alta_medica", "dt_alta_admin",
             "dt_ingreso", "dt_salida", "tipo_habitacion", "p_dpto"]]
        .sort_values("dt_alta_medica", na_position="last")
        .groupby("episodio_id", as_index=False)
        .last()
    )

    # 3. Facturas: Hosffa
    ffa = ffa.copy()
    ffa["episodio_id"] = make_episodio_id(ffa["area"], ffa["folio"], ffa["ext"])
    ffa["dt_factura"] = parse_fecha(ffa["f_fac"])
    ffa_agg = ffa.groupby("episodio_id").agg(
        fecha_primera_factura=("dt_factura", "min"),
        fecha_ultima_factura=("dt_factura", "max"),
        num_facturas=("num_fac", "count"),
        monto_total_facturas=("total", "sum"),
    ).reset_index()

    # 4. Transacciones
    trans = trans.copy()
    trans["episodio_id"] = make_episodio_id(trans["area"], trans["folio"], trans["ext"])
    trans["dt_trans"] = parse_fecha(trans["fecha"])
    trans["cancelada_flag"] = trans["cancelada"].astype(str).str.strip() == "1"
    trans["negativa_flag"] = trans["naturaleza"].astype(str).str.strip() == "1"
    trans["dep_sol"] = trans["dep_sol"].astype(str).str.strip()
    trans["num_req"] = trans["num_req"].astype(str).str.strip()

    trans_agg = trans.groupby("episodio_id").agg(
        fecha_primera_transaccion=("dt_trans", "min"),
        fecha_ultima_transaccion=("dt_trans", "max"),
        num_transacciones=("consecutivo", "count"),
        num_transacciones_negativas=("negativa_flag", "sum"),
        num_transacciones_canceladas=("cancelada_flag", "sum"),
        monto_total_transacciones=("total", "sum"),
        dep_sol_principal=("dep_sol", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
    ).reset_index()
    trans_agg["dep_sol_principal"] = trans_agg["dep_sol_principal"].astype(str).str.strip()

    # 5. Requisiciones: Hostransacciones tipo R + Hosder
    ep_ids_set = set(fol_ep["episodio_id"])
    trans_r = trans[trans["tipo"].astype(str).str.strip() == "R"].copy()
    trans_r = trans_r[trans_r["episodio_id"].isin(ep_ids_set)]
    req_unique = trans_r.drop_duplicates(subset=["episodio_id", "dep_sol", "num_req"])

    der = der.copy()
    der["dep_sol"] = der["dep_sol"].astype(str).str.strip()
    der["num_req"] = der["num_req"].astype(str).str.strip()
    der_agg = der.groupby(["dep_sol", "num_req"], as_index=False).agg(
        monto_req=("sub_total", "sum"),
        num_items=("par", "count"),
    )
    req_detail = req_unique.merge(der_agg, on=["dep_sol", "num_req"], how="left")
    req_agg = req_detail.groupby("episodio_id").agg(
        fecha_primera_requisicion=("dt_trans", "min"),
        fecha_ultima_requisicion=("dt_trans", "max"),
        num_requisiciones=("num_req", "count"),
        monto_total_requisiciones=("monto_req", "sum"),
    ).reset_index()

    # 6. Construcción tabla final
    ep = fol_ep[["episodio_id", "folio", "ext", "area", "paciente_id",
                 "dt_apertura", "dt_cierre", "f_status"]].copy()
    ep.rename(columns={"area": "area_principal", "f_status": "estatus_folio"}, inplace=True)

    ep = ep.merge(pac_ep, on="episodio_id", how="left")
    ep = ep.merge(ffa_agg, on="episodio_id", how="left")
    ep = ep.merge(trans_agg, on="episodio_id", how="left")
    ep = ep.merge(req_agg, on="episodio_id", how="left")

    ep["tiempo_alta_admin_horas"] = horas_entre(ep["dt_alta_medica"], ep["dt_alta_admin"])
    ep["tiempo_ciclo_facturacion_horas"] = horas_entre(ep["fecha_primera_factura"], ep["fecha_ultima_factura"])

    ep["tiene_alta_medica"] = ep["dt_alta_medica"].notna()
    ep["tiene_alta_administrativa"] = ep["dt_alta_admin"].notna()
    ep["tiene_tiempo_calculable"] = ep["tiempo_alta_admin_horas"].notna()

    def clasificar_estatus(row):
        am = row["tiene_alta_medica"]
        aa = row["tiene_alta_administrativa"]
        tc = row["tiene_tiempo_calculable"]
        if am and aa and tc:
            return "Cerrado con tiempo calculable"
        if am and aa and not tc:
            return "Cerrado con datos inconsistentes"
        if aa and not am:
            return "Cerrado sin alta médica"
        if am and not aa:
            return "Pendiente"
        return "Sin trazabilidad"

    ep["estatus_cierre"] = ep.apply(clasificar_estatus, axis=1)
    ep["nivel_retraso"] = ep["tiempo_alta_admin_horas"].apply(clasificar_retraso)
    ep["grupo_facturacion"] = np.where(
        ep["num_facturas"].fillna(0) == 0,
        "Sin factura",
        np.where(ep["num_facturas"] == 1, "1 factura", "Múltiples facturas")
    )

    fecha_referencia = ep["dt_alta_medica"].max()
    ep["horas_desde_alta_medica"] = np.where(
        ep["tiene_alta_medica"] & ~ep["tiene_alta_administrativa"],
        (fecha_referencia - ep["dt_alta_medica"]).dt.total_seconds() / 3600,
        np.nan,
    )
    ep["horas_desde_alta_medica"] = ep["horas_desde_alta_medica"].clip(lower=0)
    ep["nivel_prioridad"] = ep["horas_desde_alta_medica"].apply(clasificar_prioridad)

    for col in ["num_facturas", "num_transacciones", "num_transacciones_negativas",
                "num_transacciones_canceladas", "num_requisiciones"]:
        ep[col] = ep[col].fillna(0).astype(int)
    for col in ["monto_total_facturas", "monto_total_transacciones", "monto_total_requisiciones"]:
        ep[col] = ep[col].fillna(0).round(2)

    ep["tiene_facturas"] = ep["num_facturas"] > 0
    ep["tiene_transacciones"] = ep["num_transacciones"] > 0
    ep["tiene_requisiciones"] = ep["num_requisiciones"] > 0

    trans_p99 = ep.loc[ep["num_transacciones"] > 0, "num_transacciones"].quantile(0.99)
    req_p99 = ep.loc[ep["num_requisiciones"] > 0, "num_requisiciones"].quantile(0.99)
    ep["outlier_transacciones"] = ep["num_transacciones"] > trans_p99
    ep["outlier_requisiciones"] = ep["num_requisiciones"] > req_p99
    ep["outlier_nota"] = ""
    ep.loc[ep["outlier_transacciones"], "outlier_nota"] += "transacciones_atipicas;"
    ep.loc[ep["outlier_requisiciones"], "outlier_nota"] += "requisiciones_atipicas;"
    ep["outlier_nota"] = ep["outlier_nota"].str.rstrip(";")

    ep["tipo_habitacion"] = ep["tipo_habitacion"].fillna("Desconocido").replace({"": "Desconocido", "nan": "Desconocido"})
    ep["ext"] = pd.to_numeric(ep["ext"], errors="coerce").astype("Int64").astype(str)
    ep["ext"] = ep["ext"].replace("<NA>", "").str.strip()

    for col in ["episodio_id", "paciente_id", "folio", "paciente_id_pac", "estatus_folio",
                "area_principal", "tipo_habitacion", "dep_sol_principal"]:
        if col in ep.columns:
            ep[col] = ep[col].fillna("").astype(str).str.strip()

    date_dt_cols = ["dt_apertura", "dt_cierre", "dt_ingreso", "dt_salida", "dt_alta_medica", "dt_alta_admin"]
    for c in date_dt_cols:
        if c in ep.columns:
            ep[c] = ep[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(ep[c].notna(), other="")

    date_str_cols = ["fecha_primera_factura", "fecha_ultima_factura", "fecha_primera_transaccion",
                     "fecha_ultima_transaccion", "fecha_primera_requisicion", "fecha_ultima_requisicion"]
    for c in date_str_cols:
        if c in ep.columns:
            temp = pd.to_datetime(ep[c], errors="coerce")
            ep[c] = temp.dt.strftime("%Y-%m-%d %H:%M:%S").where(temp.notna(), other="")

    ep.rename(columns={
        "paciente_id_pac": "paciente_exp",
        "dt_apertura": "fecha_apertura_folio",
        "dt_cierre": "fecha_cierre_folio",
        "dt_ingreso": "fecha_ingreso",
        "dt_salida": "fecha_salida",
        "dt_alta_medica": "fecha_alta_medica",
        "dt_alta_admin": "fecha_alta_administrativa",
        "p_dpto": "departamento",
    }, inplace=True)

    for c in ["fecha_apertura_folio", "fecha_cierre_folio", "fecha_ingreso", "fecha_salida",
              "fecha_alta_medica", "fecha_alta_administrativa"] + date_str_cols:
        if c in ep.columns:
            ep[c] = ep[c].astype(str).replace({"nan": "", "NaT": "", "None": ""})

    ep_out = ep[[
        "episodio_id", "paciente_id", "paciente_exp", "folio", "ext",
        "area_principal", "estatus_folio", "estatus_cierre",
        "tiene_alta_medica", "tiene_alta_administrativa", "tiene_tiempo_calculable",
        "tiene_facturas", "tiene_transacciones", "tiene_requisiciones",
        "fecha_apertura_folio", "fecha_cierre_folio", "fecha_ingreso", "fecha_salida",
        "fecha_alta_medica", "fecha_alta_administrativa",
        "tiempo_alta_admin_horas", "nivel_retraso",
        "horas_desde_alta_medica", "nivel_prioridad",
        "fecha_primera_factura", "fecha_ultima_factura",
        "num_facturas", "monto_total_facturas",
        "tiempo_ciclo_facturacion_horas", "grupo_facturacion",
        "fecha_primera_transaccion", "fecha_ultima_transaccion",
        "num_transacciones", "num_transacciones_negativas",
        "num_transacciones_canceladas", "monto_total_transacciones",
        "dep_sol_principal",
        "fecha_primera_requisicion", "fecha_ultima_requisicion",
        "num_requisiciones", "monto_total_requisiciones",
        "tipo_habitacion", "departamento",
        "outlier_transacciones", "outlier_requisiciones", "outlier_nota",
    ]].copy()

    output_path = OUTPUT_DIR / "episodios_dashboard.csv"
    ep_out.to_csv(output_path, index=False)
    print(f"Listo: {output_path} -> {len(ep_out):,} episodios")


if __name__ == "__main__":
    main()
