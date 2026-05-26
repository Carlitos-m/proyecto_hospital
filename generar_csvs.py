import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("Data/raw")
OUT = Path("Data/output")
OUT.mkdir(exist_ok=True)
ENC = "utf-8-sig"

# ── 1. CARGA ──────────────────────────────────────────────────────────────────
print("Cargando datos...")
hospac = pd.read_csv(RAW / "Hospac.csv", low_memory=False)
hosffa = pd.read_csv(RAW / "Hosffa.csv", low_memory=False)
hosreq = pd.read_csv(RAW / "Hosreq.csv", low_memory=False)
hosfol = pd.read_csv(RAW / "Hosfol.csv", low_memory=False)
hosder = pd.read_csv(RAW / "Hosder.csv", low_memory=False)
# Hostransacciones no esta disponible como CSV; se usa Hosfol como proxy.

# ── 2. FUNCIONES DE FECHA ─────────────────────────────────────────────────────
def to_date(col):
    s = col.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan, "0": np.nan, "00000000": np.nan})
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

def to_datetime_col(date_col, time_col):
    d = date_col.astype(str).str.strip()
    d = d.replace({"nan": np.nan, "": np.nan, "0": np.nan, "00000000": np.nan})
    t = time_col.astype(str).str.strip()
    t = t.replace({"nan": np.nan, "": np.nan})
    t = t.str.zfill(6)
    combined = d.str.cat(t, na_rep="")
    mask_bad = d.isna() | t.isna() | (d == "") | (t == "")
    combined[mask_bad] = np.nan
    return pd.to_datetime(combined, format="%Y%m%d%H%M%S", errors="coerce")

hospac["fecha_alta_medica"]         = to_datetime_col(hospac["FechaAltaMedica"],         hospac["HoraAltaMedica"])
hospac["fecha_alta_administrativa"] = to_datetime_col(hospac["FechaAltaAdministrativa"], hospac["HoraAltaAdministrativa"])
hospac["p_res_fec_dt"]              = to_date(hospac["p_res_fec"])

hosffa["f_fac_dt"]     = to_date(hosffa["f_fac"])
hosreq["fec_sol_dt"]   = to_date(hosreq["fec_sol"])
hosfol["f_fec_ape_dt"] = to_date(hosfol["f_fec_ape"])

# ── 3. LLAVES DE JOIN (todo como string) ──────────────────────────────────────
hospac["folio"] = hospac["p_fol_cto"].astype(str).str.strip()
hospac["ext"]   = hospac["p_fol_ext"].astype(str).str.strip()
hosffa["folio"] = hosffa["folio"].astype(str).str.strip()
hosffa["ext"]   = hosffa["ext"].astype(str).str.strip()
hosreq["folio"] = hosreq["folio"].astype(str).str.strip()
hosreq["ext"]   = hosreq["folioext"].astype(str).str.strip()
hosfol["folio"] = hosfol["f_folio"].astype(str).str.strip()
hosfol["ext"]   = hosfol["f_folio_ext"].astype(str).str.strip()

# ── 4. DEDUPLICAR HOSPAC (un episodio por folio+ext) ──────────────────────────
hospac_ep = (
    hospac
    .sort_values("p_res_fec_dt", ascending=False, na_position="last")
    .drop_duplicates(subset=["folio", "ext"], keep="first")
    .reset_index(drop=True)
)
print(f"Hospac deduplicado: {len(hospac_ep)} episodios unicos")

# ── 5. AGRUPACIONES ───────────────────────────────────────────────────────────

# Facturas
fac_agg = hosffa.groupby(["folio", "ext"]).agg(
    num_facturas          = ("num_fac",  "count"),
    fecha_primera_factura = ("f_fac_dt", "min"),
    fecha_ultima_factura  = ("f_fac_dt", "max"),
    monto_total_facturas  = ("total",    "sum"),
).reset_index()
fac_agg["tiempo_ciclo_facturacion_horas"] = (
    (fac_agg["fecha_ultima_factura"] - fac_agg["fecha_primera_factura"])
    .dt.total_seconds() / 3600
).round(2)
fac_agg["grupo_facturacion"] = fac_agg["num_facturas"].apply(
    lambda n: "1 factura" if n == 1 else "2+ facturas"
)

# Requisiciones
req_agg = hosreq.groupby(["folio", "ext"]).agg(
    num_requisiciones         = ("num_req",    "nunique"),
    monto_total_requisiciones = ("monto",      "sum"),
    fecha_primera_requisicion = ("fec_sol_dt", "min"),
    fecha_ultima_requisicion  = ("fec_sol_dt", "max"),
    dep_sol_principal         = ("dep_sol", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
).reset_index()

# Transacciones (proxy desde Hosfol)
fol_agg = hosfol.groupby(["folio", "ext"]).agg(
    monto_total_transacciones = ("f_cargos",    "sum"),
    num_transacciones         = ("f_folio",     "count"),
    fecha_primera_transaccion = ("f_fec_ape_dt","min"),
    fecha_ultima_transaccion  = ("f_fec_ape_dt","max"),
).reset_index()
fol_agg["num_transacciones_negativas"]  = 0
fol_agg["num_transacciones_canceladas"] = 0

# Promedio items por requisicion (Hosder: una fila por linea de detalle)
_promedio_items = round(hosder.groupby("num_req").size().mean(), 2)

# ── 6. TABLA BASE DE EPISODIOS ─────────────────────────────────────────────────
ep = hospac_ep[[
    "p_num_exp", "folio", "ext",
    "p_area", "fecha_alta_medica", "fecha_alta_administrativa",
    "CuartoEntrada", "TipoAtencion",
]].copy()

ep["episodio_id"] = ep["folio"] + "_" + ep["ext"]

ep.rename(columns={
    "p_num_exp":     "paciente_id",
    "p_area":        "area_principal",
    "CuartoEntrada": "tipo_habitacion",
    "TipoAtencion":  "tipo_atencion",
}, inplace=True)

ep = (ep
    .merge(fac_agg, on=["folio", "ext"], how="left")
    .merge(req_agg, on=["folio", "ext"], how="left")
    .merge(fol_agg, on=["folio", "ext"], how="left")
)

# Tiempo entre alta medica y alta administrativa
ep["tiempo_alta_admin_horas"] = (
    (ep["fecha_alta_administrativa"] - ep["fecha_alta_medica"])
    .dt.total_seconds() / 3600
).round(2)
# Negativos = error de calidad de datos
ep.loc[ep["tiempo_alta_admin_horas"] < 0, "tiempo_alta_admin_horas"] = np.nan

# Estatus de cierre (3 valores)
def calc_estatus(row):
    if pd.notna(row["fecha_alta_administrativa"]):
        return "Cerrado"
    if pd.notna(row["fecha_alta_medica"]):
        return "Pendiente"
    return "Sin trazabilidad suficiente"

ep["estatus_cierre"] = ep.apply(calc_estatus, axis=1)

# Nivel de retraso (umbrales nuevos)
def calc_nivel_retraso(h):
    if pd.isna(h):  return np.nan
    if h <= 24:     return "Rapido"
    if h <= 72:     return "Moderado"
    if h <= 168:    return "Alto"
    return "Critico"

ep["nivel_retraso"] = ep["tiempo_alta_admin_horas"].apply(calc_nivel_retraso)

# Grupo facturacion
ep["grupo_facturacion"] = ep["grupo_facturacion"].fillna("Sin facturas")

# Relleno numerico
num_cols = [
    "num_facturas", "num_requisiciones", "num_transacciones",
    "num_transacciones_negativas", "num_transacciones_canceladas",
    "monto_total_requisiciones", "monto_total_transacciones", "monto_total_facturas",
]
ep[num_cols] = ep[num_cols].fillna(0)
ep["tiempo_ciclo_facturacion_horas"] = ep["tiempo_ciclo_facturacion_horas"].fillna(0)

# FECHA_REF = max de fecha_alta_medica (NO pd.Timestamp.now)
FECHA_REF = ep["fecha_alta_medica"].max()
print(f"Fecha referencia: {FECHA_REF}")

ep["horas_desde_alta_medica"] = ep.apply(
    lambda r: round((FECHA_REF - r["fecha_alta_medica"]).total_seconds() / 3600, 2)
    if pd.notna(r["fecha_alta_medica"]) and pd.isna(r["fecha_alta_administrativa"])
    else np.nan,
    axis=1,
)

def calc_nivel_prioridad(h):
    if pd.isna(h):  return np.nan
    if h <= 24:     return "Baja"
    if h <= 48:     return "Media"
    if h <= 72:     return "Alta"
    return "Critica"

ep["nivel_prioridad"] = ep["horas_desde_alta_medica"].apply(calc_nivel_prioridad)

# ── 7. episodios_dashboard.csv ────────────────────────────────────────────────
ep.to_csv(OUT / "episodios_dashboard.csv", index=False, encoding=ENC)
print("OK episodios_dashboard.csv")

total = len(ep)

# ── 8. kpis_inicio.csv ────────────────────────────────────────────────────────
# Seccion A: tarjetas KPI en formato largo (kpi, valor, descripcion)
kpi_rows = [
    ("total_episodios",
     int(total),
     "Total de episodios en el sistema"),
    ("episodios_con_alta_medica",
     int(ep["fecha_alta_medica"].notna().sum()),
     "Episodios con fecha de alta medica registrada"),
    ("episodios_con_alta_administrativa",
     int(ep["fecha_alta_administrativa"].notna().sum()),
     "Episodios con fecha de alta administrativa registrada"),
    ("episodios_pendientes",
     int(ep["estatus_cierre"].eq("Pendiente").sum()),
     "Episodios con alta medica pero sin alta administrativa"),
    ("episodios_sin_trazabilidad",
     int(ep["estatus_cierre"].eq("Sin trazabilidad suficiente").sum()),
     "Episodios sin fecha de alta medica ni administrativa"),
    ("tiempo_promedio_alta_admin_horas",
     round(ep["tiempo_alta_admin_horas"].mean(), 2),
     "Promedio de horas entre alta medica y alta administrativa (excluye negativos)"),
    ("tiempo_mediana_alta_admin_horas",
     round(ep["tiempo_alta_admin_horas"].median(), 2),
     "Mediana de horas entre alta medica y alta administrativa"),
    ("porcentaje_retraso_24h",
     round(ep["tiempo_alta_admin_horas"].gt(24).sum() / total * 100, 2),
     "Porcentaje de episodios cerrados con mas de 24h de retraso"),
    ("porcentaje_casos_criticos",
     round(ep["nivel_retraso"].eq("Critico").sum() / total * 100, 2),
     "Porcentaje de episodios con nivel de retraso Critico (>168h)"),
]
kpi_cards = pd.DataFrame(kpi_rows, columns=["kpi", "valor", "descripcion"])
kpi_cards["tipo_fila"]        = "kpi_global"
kpi_cards["categoria"]        = np.nan
kpi_cards["conteo_episodios"] = np.nan

# Seccion B: por estatus de cierre (dona)
por_estatus = (
    ep.groupby("estatus_cierre")
    .agg(conteo_episodios=("episodio_id", "count"))
    .reset_index()
    .rename(columns={"estatus_cierre": "categoria"})
)
por_estatus["tipo_fila"]   = "por_estatus"
por_estatus["kpi"]         = np.nan
por_estatus["valor"]       = np.nan
por_estatus["descripcion"] = np.nan

# Seccion C: por nivel de retraso (barras)
_orden_nr = {"Rapido": 0, "Moderado": 1, "Alto": 2, "Critico": 3}
por_nivel = (
    ep.groupby("nivel_retraso", dropna=False)
    .agg(conteo_episodios=("episodio_id", "count"))
    .reset_index()
    .rename(columns={"nivel_retraso": "categoria"})
)
por_nivel["tipo_fila"]   = "por_nivel_retraso"
por_nivel["_ord"]        = por_nivel["categoria"].map(_orden_nr)
por_nivel = por_nivel.sort_values("_ord").drop(columns=["_ord"])
por_nivel["kpi"]         = np.nan
por_nivel["valor"]       = np.nan
por_nivel["descripcion"] = np.nan

# Seccion D: distribucion de tiempos (histograma con umbrales nuevos)
bins_t = [-float("inf"), 0, 12, 24, 48, 72, 168, float("inf")]
labs_t = ["Sin alta medica", "0-12h", "12-24h", "24-48h", "48-72h", "72-168h", ">168h"]
ep["_rango_t"] = pd.cut(ep["tiempo_alta_admin_horas"], bins=bins_t, labels=labs_t, right=True)
por_rango_t = (
    ep.groupby("_rango_t", observed=True)
    .agg(conteo_episodios=("episodio_id", "count"))
    .reset_index()
    .rename(columns={"_rango_t": "categoria"})
)
por_rango_t["tipo_fila"]   = "rango_tiempo"
por_rango_t["kpi"]         = np.nan
por_rango_t["valor"]       = np.nan
por_rango_t["descripcion"] = np.nan
ep.drop(columns=["_rango_t"], inplace=True)

kpis = pd.concat([kpi_cards, por_estatus, por_nivel, por_rango_t], ignore_index=True)
kpis.to_csv(OUT / "kpis_inicio.csv", index=False, encoding=ENC)
print("OK kpis_inicio.csv")

# ── 9. cuellos_botella.csv ────────────────────────────────────────────────────
def etapa(df, col_ini, col_fin, nombre):
    diff = (df[col_fin] - df[col_ini]).dt.total_seconds() / 3600
    diff = diff[diff > 0]
    return {
        "etapa":                 nombre,
        "tiempo_promedio_horas": round(diff.mean(), 2),
        "tiempo_mediana_horas":  round(diff.median(), 2),
        "num_episodios_validos": int(diff.notna().sum()),
    }

cuellos = pd.DataFrame([
    etapa(ep, "fecha_alta_medica",        "fecha_primera_factura",     "Alta medica -> Primera factura"),
    etapa(ep, "fecha_primera_factura",    "fecha_ultima_factura",      "Primera factura -> Ultima factura"),
    etapa(ep, "fecha_primera_transaccion","fecha_ultima_transaccion",  "Primera transaccion -> Ultima transaccion"),
    etapa(ep, "fecha_primera_requisicion","fecha_ultima_requisicion",  "Primera requisicion -> Ultima requisicion"),
    etapa(ep, "fecha_ultima_factura",     "fecha_alta_administrativa", "Ultima factura -> Alta administrativa"),
])
cuellos.to_csv(OUT / "cuellos_botella.csv", index=False, encoding=ENC)
print("OK cuellos_botella.csv")

# ── 10. facturacion_resumen.csv ───────────────────────────────────────────────
_ciclo_valido = ep["tiempo_ciclo_facturacion_horas"][ep["tiempo_ciclo_facturacion_horas"] > 0]
_idx_max_fac  = ep["num_facturas"].idxmax()

fac_kpi = pd.DataFrame([{
    "tipo_fila":                               "kpi",
    "categoria":                               "global",
    "conteo_episodios":                        total,
    "num_facturas":                            np.nan,
    "promedio_facturas_por_episodio":          round(ep["num_facturas"].mean(), 2),
    "pct_episodios_multiples_facturas":        round(ep["num_facturas"].gt(1).sum() / total * 100, 2),
    "tiempo_promedio_ciclo_facturacion_horas": round(_ciclo_valido.mean(), 2),
    "max_facturas_episodio":                   int(ep["num_facturas"].max()),
    "episodio_max_facturas":                   ep.at[_idx_max_fac, "episodio_id"],
    "tiempo_promedio_alta_admin_horas":        round(ep["tiempo_alta_admin_horas"].mean(), 2),
    "episodio_id":                             np.nan,
    "folio":                                   np.nan,
    "ext":                                     np.nan,
    "fecha_primera_factura":                   np.nan,
    "fecha_ultima_factura":                    np.nan,
    "tiempo_ciclo_facturacion_horas":          np.nan,
    "nivel_retraso":                           np.nan,
}])

_fac_nulls = [
    "promedio_facturas_por_episodio", "pct_episodios_multiples_facturas",
    "max_facturas_episodio", "episodio_max_facturas",
    "episodio_id", "folio", "ext",
    "fecha_primera_factura", "fecha_ultima_factura",
    "tiempo_ciclo_facturacion_horas", "nivel_retraso",
]

# Por numero de facturas
por_num = ep.groupby("num_facturas").agg(
    conteo_episodios                        = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas        = ("tiempo_alta_admin_horas", "mean"),
    tiempo_promedio_ciclo_facturacion_horas = ("tiempo_ciclo_facturacion_horas", "mean"),
).reset_index().round(2)
por_num["tipo_fila"] = "por_num_facturas"
por_num["categoria"] = por_num["num_facturas"].astype(str) + " factura(s)"
por_num[_fac_nulls]  = np.nan

# Por grupo (1 factura vs 2+ facturas)
por_grupo = (
    ep[ep["grupo_facturacion"] != "Sin facturas"]
    .groupby("grupo_facturacion")
    .agg(
        conteo_episodios                        = ("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas        = ("tiempo_alta_admin_horas", "mean"),
        tiempo_promedio_ciclo_facturacion_horas = ("tiempo_ciclo_facturacion_horas", "mean"),
    )
    .reset_index()
    .round(2)
)
por_grupo["tipo_fila"] = "por_grupo"
por_grupo.rename(columns={"grupo_facturacion": "categoria"}, inplace=True)
por_grupo["num_facturas"] = np.nan
por_grupo[_fac_nulls]     = np.nan

# Distribucion ciclo facturacion
bins_c = [0, 12, 24, 48, 72, float("inf")]
labs_c = ["0-12h", "12-24h", "24-48h", "48-72h", ">72h"]
ep["_rango_c"] = pd.cut(ep["tiempo_ciclo_facturacion_horas"], bins=bins_c, labels=labs_c, right=True)
rango_ciclo = (
    ep.groupby("_rango_c", observed=True)
    .agg(conteo_episodios=("episodio_id", "count"))
    .reset_index()
    .rename(columns={"_rango_c": "categoria"})
)
rango_ciclo["tipo_fila"]   = "rango_ciclo"
rango_ciclo["num_facturas"] = np.nan
rango_ciclo[["tiempo_promedio_alta_admin_horas",
             "tiempo_promedio_ciclo_facturacion_horas"]] = np.nan
rango_ciclo[_fac_nulls] = np.nan
ep.drop(columns=["_rango_c"], inplace=True)

# Top 20 episodios con mas facturas
top_fac = (
    ep[ep["num_facturas"] > 0]
    .nlargest(20, "num_facturas")
    [["episodio_id", "folio", "ext", "num_facturas",
      "fecha_primera_factura", "fecha_ultima_factura",
      "tiempo_ciclo_facturacion_horas", "tiempo_alta_admin_horas", "nivel_retraso"]]
    .copy()
)
top_fac["tipo_fila"] = "top_episodios"
top_fac["categoria"] = np.nan
top_fac[["conteo_episodios",
         "promedio_facturas_por_episodio", "pct_episodios_multiples_facturas",
         "max_facturas_episodio", "episodio_max_facturas",
         "tiempo_promedio_alta_admin_horas",
         "tiempo_promedio_ciclo_facturacion_horas"]] = np.nan

fac_res = pd.concat([fac_kpi, por_num, por_grupo, rango_ciclo, top_fac], ignore_index=True)
fac_res.to_csv(OUT / "facturacion_resumen.csv", index=False, encoding=ENC)
print("OK facturacion_resumen.csv")

# ── 11. facturacion_grupo.csv (NUEVO) ─────────────────────────────────────────
fac_grupo = (
    ep.groupby("grupo_facturacion")
    .agg(
        conteo_episodios                        = ("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas        = ("tiempo_alta_admin_horas", "mean"),
        tiempo_mediana_alta_admin_horas         = ("tiempo_alta_admin_horas", "median"),
        tiempo_promedio_ciclo_facturacion_horas = ("tiempo_ciclo_facturacion_horas", "mean"),
        monto_total_facturas                    = ("monto_total_facturas", "sum"),
        promedio_facturas_por_episodio          = ("num_facturas", "mean"),
    )
    .reset_index()
    .round(2)
)
fac_grupo["pct_del_total"] = (fac_grupo["conteo_episodios"] / total * 100).round(2)
fac_grupo.to_csv(OUT / "facturacion_grupo.csv", index=False, encoding=ENC)
print("OK facturacion_grupo.csv")

# ── 12. transacciones_resumen.csv ─────────────────────────────────────────────
trans_kpi = pd.DataFrame([{
    "tipo_fila":                           "kpi",
    "categoria":                           "global",
    "conteo_episodios":                    total,
    "promedio_transacciones_por_episodio": round(ep["num_transacciones"].mean(), 2),
    "pct_con_transacciones_negativas":     round(ep["num_transacciones_negativas"].gt(0).sum() / total * 100, 2),
    "pct_con_cancelaciones":               round(ep["num_transacciones_canceladas"].gt(0).sum() / total * 100, 2),
    "monto_total_transacciones":           round(ep["monto_total_transacciones"].sum(), 2),
    "tiempo_promedio_alta_admin_horas":    round(ep["tiempo_alta_admin_horas"].mean(), 2),
    "fuente":                              "Proxy Hosfol (Hostransacciones no disponible como CSV)",
}])

ep["_tiene_neg"] = ep["num_transacciones_negativas"].gt(0).map(
    {True: "Con negativas", False: "Sin negativas"}
)
por_negativas = (
    ep.groupby("_tiene_neg")
    .agg(
        conteo_episodios                = ("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
        monto_total_transacciones       = ("monto_total_transacciones", "sum"),
    )
    .reset_index()
    .round(2)
)
por_negativas["tipo_fila"] = "por_negativas"
por_negativas.rename(columns={"_tiene_neg": "categoria"}, inplace=True)
por_negativas[["promedio_transacciones_por_episodio",
               "pct_con_transacciones_negativas",
               "pct_con_cancelaciones", "fuente"]] = np.nan

bins_tr   = [0, 1, 3, 5, 10, float("inf")]
labels_tr = ["0", "1-3", "3-5", "5-10", ">10"]
ep["_rango_tr"] = pd.cut(ep["num_transacciones"], bins=bins_tr, labels=labels_tr, right=True)
por_rango_tr = (
    ep.groupby("_rango_tr", observed=True)
    .agg(
        conteo_episodios                = ("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
    )
    .reset_index()
    .round(2)
)
por_rango_tr["tipo_fila"] = "rango_transacciones"
por_rango_tr.rename(columns={"_rango_tr": "categoria"}, inplace=True)
por_rango_tr[["promedio_transacciones_por_episodio", "pct_con_transacciones_negativas",
              "pct_con_cancelaciones", "monto_total_transacciones", "fuente"]] = np.nan

trans_res = pd.concat([trans_kpi, por_negativas, por_rango_tr], ignore_index=True)
trans_res.to_csv(OUT / "transacciones_resumen.csv", index=False, encoding=ENC)
print("OK transacciones_resumen.csv")
ep.drop(columns=["_tiene_neg", "_rango_tr"], inplace=True)

# ── 13. requisiciones_resumen.csv ─────────────────────────────────────────────
total_con_req = int(ep["num_requisiciones"].gt(0).sum())

req_kpi = pd.DataFrame([{
    "tipo_fila":                           "kpi",
    "categoria":                           "global",
    "dep_sol_principal":                   np.nan,
    "conteo_episodios":                    total,
    "conteo_episodios_con_req":            total_con_req,
    "num_requisiciones":                   np.nan,
    "monto_total_requisiciones":           round(ep["monto_total_requisiciones"].sum(), 2),
    "tiempo_promedio_alta_admin_horas":    round(
        ep.loc[ep["num_requisiciones"] > 0, "tiempo_alta_admin_horas"].mean(), 2
    ),
    "nivel_retraso_predominante":          (
        ep["nivel_retraso"].mode().iloc[0] if ep["nivel_retraso"].notna().any() else np.nan
    ),
    "promedio_requisiciones_por_episodio": round(
        ep.loc[ep["num_requisiciones"] > 0, "num_requisiciones"].mean(), 2
    ),
    "pct_con_requisiciones":               round(total_con_req / total * 100, 2),
    "promedio_items_por_requisicion":      _promedio_items,
    "episodio_id":                         np.nan,
    "folio":                               np.nan,
    "ext":                                 np.nan,
    "descripcion":                         "KPI globales de requisiciones",
}])

_req_nulls = [
    "promedio_requisiciones_por_episodio",
    "pct_con_requisiciones",
    "promedio_items_por_requisicion",
]

# Por departamento solicitante (ordenado de mayor a menor retraso)
por_dep = (
    ep[ep["dep_sol_principal"].notna()]
    .groupby("dep_sol_principal")
    .agg(
        conteo_episodios                = ("episodio_id", "count"),
        num_requisiciones               = ("num_requisiciones", "sum"),
        monto_total_requisiciones       = ("monto_total_requisiciones", "sum"),
        tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
        nivel_retraso_predominante      = (
            "nivel_retraso", lambda x: x.mode().iloc[0] if x.notna().any() else np.nan
        ),
    )
    .reset_index()
    .round(2)
    .sort_values("tiempo_promedio_alta_admin_horas", ascending=False)
)
por_dep["tipo_fila"]             = "por_departamento"
por_dep.rename(columns={"dep_sol_principal": "categoria"}, inplace=True)
por_dep["dep_sol_principal"]     = por_dep["categoria"]
por_dep["conteo_episodios_con_req"] = np.nan
por_dep["episodio_id"]           = np.nan
por_dep["folio"]                 = np.nan
por_dep["ext"]                   = np.nan
por_dep["descripcion"]           = np.nan
por_dep[_req_nulls]              = np.nan

# Rango requisiciones vs tiempo de alta
bins_r = [-0.5, 0, 2, 5, 10, float("inf")]
labs_r = ["0", "1-2", "3-5", "6-10", ">10"]
ep["_rango_r"] = pd.cut(ep["num_requisiciones"], bins=bins_r, labels=labs_r, right=True)
por_rango_r = (
    ep.groupby("_rango_r", observed=True)
    .agg(
        conteo_episodios                = ("episodio_id", "count"),
        tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
        monto_total_requisiciones       = ("monto_total_requisiciones", "sum"),
    )
    .reset_index()
    .round(2)
    .rename(columns={"_rango_r": "categoria"})
)
por_rango_r["tipo_fila"]             = "rango_requisiciones"
por_rango_r["dep_sol_principal"]     = np.nan
por_rango_r["num_requisiciones"]     = np.nan
por_rango_r["nivel_retraso_predominante"] = np.nan
por_rango_r["conteo_episodios_con_req"]   = np.nan
por_rango_r["episodio_id"]           = np.nan
por_rango_r["folio"]                 = np.nan
por_rango_r["ext"]                   = np.nan
por_rango_r["descripcion"]           = np.nan
por_rango_r[_req_nulls]              = np.nan
ep.drop(columns=["_rango_r"], inplace=True)

# Top 20 episodios con mas requisiciones (sin duplicados)
top_req = (
    ep[ep["num_requisiciones"] > 0]
    .nlargest(20, "num_requisiciones")
    [["episodio_id", "folio", "ext", "num_requisiciones",
      "monto_total_requisiciones", "dep_sol_principal",
      "tiempo_alta_admin_horas", "nivel_retraso"]]
    .drop_duplicates(subset=["episodio_id"])
    .copy()
)
top_req["tipo_fila"]             = "top_episodios"
top_req["categoria"]             = np.nan
top_req["conteo_episodios"]      = np.nan
top_req["conteo_episodios_con_req"] = np.nan
top_req["descripcion"]           = top_req["dep_sol_principal"].apply(
    lambda d: f"Depto: {d}" if pd.notna(d) else "Sin departamento registrado"
)
top_req.rename(columns={
    "tiempo_alta_admin_horas": "tiempo_promedio_alta_admin_horas",
    "nivel_retraso":           "nivel_retraso_predominante",
}, inplace=True)
top_req[_req_nulls] = np.nan

req_res = pd.concat([req_kpi, por_dep, por_rango_r, top_req], ignore_index=True)
req_res.to_csv(OUT / "requisiciones_resumen.csv", index=False, encoding=ENC)
print("OK requisiciones_resumen.csv")

# ── 14. monitor_operativo.csv ─────────────────────────────────────────────────
monitor = (
    ep[ep["fecha_alta_medica"].notna() & ep["fecha_alta_administrativa"].isna()]
    [[
        "episodio_id", "paciente_id", "folio", "ext",
        "fecha_alta_medica", "horas_desde_alta_medica", "nivel_prioridad",
        "num_facturas", "num_transacciones", "num_requisiciones",
        "area_principal", "tipo_habitacion", "estatus_cierre",
    ]]
    .copy()
    .sort_values("horas_desde_alta_medica", ascending=False)
)
monitor["tiene_facturas"]      = monitor["num_facturas"].gt(0)
monitor["tiene_transacciones"] = monitor["num_transacciones"].gt(0)
monitor["tiene_requisiciones"] = monitor["num_requisiciones"].gt(0)
monitor["fecha_referencia"]    = FECHA_REF

monitor.to_csv(OUT / "monitor_operativo.csv", index=False, encoding=ENC)
print("OK monitor_operativo.csv")

print(f"\nListo. 8 CSVs en {OUT.resolve()}")
