import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("Data/raw")
OUT = Path("Data/output")
OUT.mkdir(exist_ok=True)

# ── 1. CARGA ──────────────────────────────────────────────────────────────────
print("Cargando datos...")
hospac = pd.read_csv(RAW / "Hospac.csv", low_memory=False)
hosffa = pd.read_csv(RAW / "Hosffa.csv", low_memory=False)
hosreq = pd.read_csv(RAW / "Hosreq.csv", low_memory=False)
hosfol = pd.read_csv(RAW / "Hosfol.csv", low_memory=False)
hosder = pd.read_csv(RAW / "Hosder.csv", low_memory=False)
# Hostransacciones no está disponible como CSV (solo en el ZIP parquet).
# Se usa Hosfol (f_cargos / f_abonos) como proxy de movimientos por folio.

# ── 2. PARSE DE FECHAS ────────────────────────────────────────────────────────
def to_date(col):
    s = col.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan, "0": np.nan, "00000000": np.nan})
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

hospac["fecha_alta_medica"]         = to_date(hospac["FechaAltaMedica"])
hospac["fecha_alta_administrativa"] = to_date(hospac["FechaAltaAdministrativa"])
hosffa["f_fac_dt"]                  = to_date(hosffa["f_fac"])
hosreq["fec_sol_dt"]               = to_date(hosreq["fec_sol"])
hosfol["f_fec_ape_dt"]             = to_date(hosfol["f_fec_ape"])

# ── 3. LLAVES DE JOIN (todo como string para evitar type mismatch) ─────────────
hospac["folio"] = hospac["p_fol_cto"].astype(str).str.strip()
hospac["ext"]   = hospac["p_fol_ext"].astype(str).str.strip()
hosffa["folio"] = hosffa["folio"].astype(str).str.strip()
hosffa["ext"]   = hosffa["ext"].astype(str).str.strip()
hosreq["folio"] = hosreq["folio"].astype(str).str.strip()
hosreq["ext"]   = hosreq["folioext"].astype(str).str.strip()
hosfol["folio"] = hosfol["f_folio"].astype(str).str.strip()
hosfol["ext"]   = hosfol["f_folio_ext"].astype(str).str.strip()

# ── 4. AGRUPACIONES POR FOLIO/EXT ─────────────────────────────────────────────

# Facturas
fac_agg = hosffa.groupby(["folio", "ext"]).agg(
    num_facturas            = ("num_fac", "count"),
    fecha_primera_factura   = ("f_fac_dt", "min"),
    fecha_ultima_factura    = ("f_fac_dt", "max"),
).reset_index()
fac_agg["tiempo_ciclo_facturacion_horas"] = (
    (fac_agg["fecha_ultima_factura"] - fac_agg["fecha_primera_factura"])
    .dt.total_seconds() / 3600
).round(2)
fac_agg["grupo_facturacion"] = fac_agg["num_facturas"].apply(
    lambda n: "Sin facturas" if n == 0 else ("Una factura" if n == 1 else "Múltiples facturas")
)

# Requisiciones
req_agg = hosreq.groupby(["folio", "ext"]).agg(
    num_requisiciones         = ("num_req", "nunique"),
    monto_total_requisiciones = ("monto", "sum"),
    fecha_primera_requisicion = ("fec_sol_dt", "min"),
    fecha_ultima_requisicion  = ("fec_sol_dt", "max"),
    dep_sol_principal         = ("dep_sol", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
).reset_index()

# Transacciones (proxy desde Hosfol)
fol_agg = hosfol.groupby(["folio", "ext"]).agg(
    monto_total_transacciones  = ("f_cargos", "sum"),
    num_transacciones          = ("f_folio", "count"),
    fecha_primera_transaccion  = ("f_fec_ape_dt", "min"),
    fecha_ultima_transaccion   = ("f_fec_ape_dt", "max"),
).reset_index()
fol_agg["num_transacciones_negativas"]  = 0
fol_agg["num_transacciones_canceladas"] = 0

# ── 5. episodios_dashboard.csv ────────────────────────────────────────────────
ep = hospac[[
    "p_res_cve_num", "p_num_exp", "folio", "ext",
    "p_area", "fecha_alta_medica", "fecha_alta_administrativa",
    "CuartoEntrada", "TipoAtencion",
]].copy()

ep.rename(columns={
    "p_res_cve_num":  "episodio_id",
    "p_num_exp":      "paciente_id",
    "p_area":         "area_principal",
    "CuartoEntrada":  "tipo_habitacion",
    "TipoAtencion":   "tipo_atencion",
}, inplace=True)

ep = (ep
    .merge(fac_agg, on=["folio", "ext"], how="left")
    .merge(req_agg, on=["folio", "ext"], how="left")
    .merge(fol_agg, on=["folio", "ext"], how="left")
)

# Campos calculados
ep["tiempo_alta_admin_horas"] = (
    (ep["fecha_alta_administrativa"] - ep["fecha_alta_medica"])
    .dt.total_seconds() / 3600
).round(2)

ep["estatus_cierre"] = ep["fecha_alta_administrativa"].apply(
    lambda x: "Cerrado" if pd.notna(x) else "Pendiente"
)

def nivel_retraso(h):
    if pd.isna(h):   return "Pendiente"
    if h <= 24:      return "Sin retraso"
    if h <= 48:      return "Retraso leve"
    if h <= 72:      return "Retraso moderado"
    return "Crítico"

ep["nivel_retraso"] = ep["tiempo_alta_admin_horas"].apply(nivel_retraso)

AHORA = pd.Timestamp.now()
ep["horas_desde_alta_medica"] = ep.apply(
    lambda r: round((AHORA - r["fecha_alta_medica"]).total_seconds() / 3600, 2)
    if pd.notna(r["fecha_alta_medica"]) and pd.isna(r["fecha_alta_administrativa"])
    else np.nan,
    axis=1,
)

def nivel_prioridad(h):
    if pd.isna(h):   return np.nan
    if h <= 24:      return "Baja"
    if h <= 48:      return "Media"
    if h <= 72:      return "Alta"
    return "Crítica"

ep["nivel_prioridad"] = ep["horas_desde_alta_medica"].apply(nivel_prioridad)

# Relleno numérico con 0
num_cols = [
    "num_facturas", "num_requisiciones", "num_transacciones",
    "num_transacciones_negativas", "num_transacciones_canceladas",
    "monto_total_requisiciones", "monto_total_transacciones",
]
ep[num_cols] = ep[num_cols].fillna(0)
ep["grupo_facturacion"]  = ep["grupo_facturacion"].fillna("Sin facturas")
ep["tiempo_ciclo_facturacion_horas"] = ep["tiempo_ciclo_facturacion_horas"].fillna(0)

ep.to_csv(OUT / "episodios_dashboard.csv", index=False)
print("OK episodios_dashboard.csv")

# ── 6. kpis_inicio.csv ────────────────────────────────────────────────────────
total = len(ep)

# Sección A: tarjetas KPI (1 fila)
kpi_global = pd.DataFrame([{
    "tipo_fila":                          "kpi_global",
    "categoria":                          np.nan,
    "conteo_episodios":                   np.nan,
    "total_episodios":                    total,
    "episodios_con_alta_medica":          int(ep["fecha_alta_medica"].notna().sum()),
    "episodios_con_alta_administrativa":  int(ep["fecha_alta_administrativa"].notna().sum()),
    "episodios_pendientes":               int(ep["estatus_cierre"].eq("Pendiente").sum()),
    "tiempo_promedio_alta_admin_horas":   round(ep["tiempo_alta_admin_horas"].mean(), 2),
    "tiempo_mediana_alta_admin_horas":    round(ep["tiempo_alta_admin_horas"].median(), 2),
    "porcentaje_retraso_24h":             round(ep["tiempo_alta_admin_horas"].gt(24).sum() / total * 100, 2),
    "porcentaje_casos_criticos":          round(ep["nivel_retraso"].eq("Crítico").sum() / total * 100, 2),
}])

_kpi_nulls = ["total_episodios", "episodios_con_alta_medica", "episodios_con_alta_administrativa",
              "episodios_pendientes", "tiempo_promedio_alta_admin_horas",
              "tiempo_mediana_alta_admin_horas", "porcentaje_retraso_24h", "porcentaje_casos_criticos"]

# Sección B: casos por estatus de cierre (dona / barras)
por_estatus = ep.groupby("estatus_cierre").agg(
    conteo_episodios=("episodio_id", "count")
).reset_index().rename(columns={"estatus_cierre": "categoria"})
por_estatus["tipo_fila"] = "por_estatus"
por_estatus[_kpi_nulls] = np.nan

# Sección C: casos por nivel de retraso (barras) — orden lógico de menor a mayor
_orden = {"Sin retraso": 0, "Retraso leve": 1, "Retraso moderado": 2, "Crítico": 3, "Pendiente": 4}
por_nivel = ep.groupby("nivel_retraso").agg(
    conteo_episodios=("episodio_id", "count")
).reset_index().rename(columns={"nivel_retraso": "categoria"})
por_nivel["tipo_fila"] = "por_nivel_retraso"
por_nivel["_ord"] = por_nivel["categoria"].map(_orden)
por_nivel = por_nivel.sort_values("_ord").drop(columns=["_ord"])
por_nivel[_kpi_nulls] = np.nan

# Sección D: distribución de tiempos de alta administrativa (barras por rangos)
bins_t  = [-float("inf"), 0, 12, 24, 48, 72, float("inf")]
labs_t  = ["Sin alta médica", "0-12h", "12-24h", "24-48h", "48-72h", ">72h"]
ep["_rango_t"] = pd.cut(ep["tiempo_alta_admin_horas"], bins=bins_t, labels=labs_t, right=True)
por_rango_t = ep.groupby("_rango_t", observed=True).agg(
    conteo_episodios=("episodio_id", "count")
).reset_index().rename(columns={"_rango_t": "categoria"})
por_rango_t["tipo_fila"] = "rango_tiempo"
por_rango_t[_kpi_nulls] = np.nan
ep.drop(columns=["_rango_t"], inplace=True)

kpis = pd.concat([kpi_global, por_estatus, por_nivel, por_rango_t], ignore_index=True)
kpis.to_csv(OUT / "kpis_inicio.csv", index=False)
print("OK kpis_inicio.csv")

# ── 7. cuellos_botella.csv ────────────────────────────────────────────────────
def etapa(df, col_ini, col_fin, nombre):
    diff = (df[col_fin] - df[col_ini]).dt.total_seconds() / 3600
    diff = diff[diff > 0]
    return {
        "etapa":                  nombre,
        "tiempo_promedio_horas":  round(diff.mean(), 2),
        "tiempo_mediana_horas":   round(diff.median(), 2),
        "num_episodios_validos":  int(diff.notna().sum()),
    }

cuellos = pd.DataFrame([
    etapa(ep, "fecha_alta_medica",       "fecha_primera_factura",    "Alta médica → Primera factura"),
    etapa(ep, "fecha_primera_factura",   "fecha_ultima_factura",     "Primera factura → Última factura"),
    etapa(ep, "fecha_primera_transaccion","fecha_ultima_transaccion", "Primera transacción → Última transacción"),
    etapa(ep, "fecha_primera_requisicion","fecha_ultima_requisicion", "Primera requisición → Última requisición"),
    etapa(ep, "fecha_ultima_factura",    "fecha_alta_administrativa", "Última factura → Alta administrativa"),
])
cuellos.to_csv(OUT / "cuellos_botella.csv", index=False)
print("OK cuellos_botella.csv")

# ── 8. facturacion_resumen.csv ────────────────────────────────────────────────
_ciclo_valido = ep["tiempo_ciclo_facturacion_horas"][ep["tiempo_ciclo_facturacion_horas"] > 0]
_idx_max_fac  = ep["num_facturas"].idxmax()

# Sección A: KPIs globales
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

_fac_nulls = ["promedio_facturas_por_episodio", "pct_episodios_multiples_facturas",
              "max_facturas_episodio", "episodio_max_facturas",
              "episodio_id", "folio", "ext",
              "fecha_primera_factura", "fecha_ultima_factura",
              "tiempo_ciclo_facturacion_horas", "nivel_retraso"]

# Sección B: tiempo promedio por número de facturas (barra)
por_num = ep.groupby("num_facturas").agg(
    conteo_episodios                        = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas        = ("tiempo_alta_admin_horas", "mean"),
    tiempo_promedio_ciclo_facturacion_horas = ("tiempo_ciclo_facturacion_horas", "mean"),
).reset_index().round(2)
por_num["tipo_fila"] = "por_num_facturas"
por_num["categoria"] = por_num["num_facturas"].astype(str) + " factura(s)"
por_num[_fac_nulls] = np.nan

# Sección C: una factura vs múltiples (barras comparativas)
por_grupo = ep.groupby("grupo_facturacion").agg(
    conteo_episodios                        = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas        = ("tiempo_alta_admin_horas", "mean"),
    tiempo_promedio_ciclo_facturacion_horas = ("tiempo_ciclo_facturacion_horas", "mean"),
).reset_index().round(2)
por_grupo["tipo_fila"] = "por_grupo"
por_grupo.rename(columns={"grupo_facturacion": "categoria"}, inplace=True)
por_grupo["num_facturas"] = np.nan
por_grupo[_fac_nulls] = np.nan

# Sección D: distribución del ciclo de facturación (histograma)
bins_c  = [0, 12, 24, 48, 72, float("inf")]
labs_c  = ["0-12h", "12-24h", "24-48h", "48-72h", ">72h"]
ep["_rango_c"] = pd.cut(ep["tiempo_ciclo_facturacion_horas"], bins=bins_c, labels=labs_c, right=True)
rango_ciclo = ep.groupby("_rango_c", observed=True).agg(
    conteo_episodios=("episodio_id", "count")
).reset_index().rename(columns={"_rango_c": "categoria"})
rango_ciclo["tipo_fila"] = "rango_ciclo"
rango_ciclo["num_facturas"] = np.nan
rango_ciclo[["tiempo_promedio_alta_admin_horas", "tiempo_promedio_ciclo_facturacion_horas"]] = np.nan
rango_ciclo[_fac_nulls] = np.nan
ep.drop(columns=["_rango_c"], inplace=True)

# Sección E: top 20 episodios con más facturas (tabla)
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
top_fac[["conteo_episodios", "promedio_facturas_por_episodio", "pct_episodios_multiples_facturas",
         "max_facturas_episodio", "episodio_max_facturas",
         "tiempo_promedio_alta_admin_horas", "tiempo_promedio_ciclo_facturacion_horas"]] = np.nan

fac_res = pd.concat([fac_kpi, por_num, por_grupo, rango_ciclo, top_fac], ignore_index=True)
fac_res.to_csv(OUT / "facturacion_resumen.csv", index=False)
print("OK facturacion_resumen.csv")

# ── 9. transacciones_resumen.csv ─────────────────────────────────────────────
# Sección A: KPIs globales
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

# Sección B: episodios con/sin transacciones negativas
ep["tiene_negativas"] = ep["num_transacciones_negativas"].gt(0).map({True: "Con negativas", False: "Sin negativas"})
por_negativas = ep.groupby("tiene_negativas").agg(
    conteo_episodios               = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
    monto_total_transacciones      = ("monto_total_transacciones", "sum"),
).reset_index().round(2)
por_negativas["tipo_fila"] = "por_negativas"
por_negativas.rename(columns={"tiene_negativas": "categoria"}, inplace=True)
por_negativas[["promedio_transacciones_por_episodio", "pct_con_transacciones_negativas",
               "pct_con_cancelaciones", "fuente"]] = np.nan

# Sección C: por número de transacciones (rangos)
bins_t   = [0, 1, 3, 5, 10, float("inf")]
labels_t = ["0", "1-3", "3-5", "5-10", ">10"]
ep["rango_trans_tmp"] = pd.cut(ep["num_transacciones"], bins=bins_t, labels=labels_t, right=True)
por_rango_t = ep.groupby("rango_trans_tmp", observed=True).agg(
    conteo_episodios               = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
).reset_index().round(2)
por_rango_t["tipo_fila"] = "rango_transacciones"
por_rango_t.rename(columns={"rango_trans_tmp": "categoria"}, inplace=True)
por_rango_t[["promedio_transacciones_por_episodio", "pct_con_transacciones_negativas",
             "pct_con_cancelaciones", "monto_total_transacciones", "fuente"]] = np.nan

trans_res = pd.concat([trans_kpi, por_negativas, por_rango_t], ignore_index=True)
trans_res.to_csv(OUT / "transacciones_resumen.csv", index=False)
print("OK transacciones_resumen.csv")
ep.drop(columns=["tiene_negativas", "rango_trans_tmp"], inplace=True)

# ── 10. requisiciones_resumen.csv ─────────────────────────────────────────────
# Promedio de ítems por requisición (desde Hosder: líneas de detalle por num_req)
_promedio_items = round(hosder.groupby("num_req").size().mean(), 2)

# Sección A: KPIs globales
req_kpi = pd.DataFrame([{
    "tipo_fila":                           "kpi",
    "categoria":                           "global",
    "dep_sol_principal":                   np.nan,
    "conteo_episodios":                    total,
    "num_requisiciones":                   np.nan,
    "monto_total_requisiciones":           round(ep["monto_total_requisiciones"].sum(), 2),
    "tiempo_promedio_alta_admin_horas":    round(ep["tiempo_alta_admin_horas"].mean(), 2),
    "nivel_retraso_predominante":          ep["nivel_retraso"].mode().iloc[0],
    "promedio_requisiciones_por_episodio": round(ep["num_requisiciones"].mean(), 2),
    "pct_con_requisiciones":               round(ep["num_requisiciones"].gt(0).sum() / total * 100, 2),
    "promedio_items_por_requisicion":      _promedio_items,
    "episodio_id":                         np.nan,
    "folio":                               np.nan,
    "ext":                                 np.nan,
}])

_req_nulls = ["promedio_requisiciones_por_episodio", "pct_con_requisiciones",
              "promedio_items_por_requisicion", "episodio_id", "folio", "ext"]

# Sección B: retraso promedio por departamento solicitante (ordenado de mayor a menor retraso)
por_dep = (
    ep[ep["dep_sol_principal"].notna()]
    .groupby("dep_sol_principal")
    .agg(
        conteo_episodios               = ("episodio_id", "count"),
        num_requisiciones              = ("num_requisiciones", "sum"),
        monto_total_requisiciones      = ("monto_total_requisiciones", "sum"),
        tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
        nivel_retraso_predominante     = ("nivel_retraso", lambda x: x.mode().iloc[0]),
    )
    .reset_index()
    .round(2)
    .sort_values("tiempo_promedio_alta_admin_horas", ascending=False)
)
por_dep["tipo_fila"] = "por_departamento"
por_dep.rename(columns={"dep_sol_principal": "categoria"}, inplace=True)
por_dep["dep_sol_principal"] = por_dep["categoria"]
por_dep[_req_nulls] = np.nan

# Sección C: requisiciones vs tiempo de alta (rangos de num_requisiciones)
bins_r  = [-0.5, 0, 2, 5, 10, float("inf")]
labs_r  = ["0", "1-2", "3-5", "6-10", ">10"]
ep["_rango_r"] = pd.cut(ep["num_requisiciones"], bins=bins_r, labels=labs_r, right=True)
por_rango_r = ep.groupby("_rango_r", observed=True).agg(
    conteo_episodios               = ("episodio_id", "count"),
    tiempo_promedio_alta_admin_horas = ("tiempo_alta_admin_horas", "mean"),
    monto_total_requisiciones      = ("monto_total_requisiciones", "sum"),
).reset_index().round(2).rename(columns={"_rango_r": "categoria"})
por_rango_r["tipo_fila"] = "rango_requisiciones"
por_rango_r["dep_sol_principal"] = np.nan
por_rango_r["num_requisiciones"] = np.nan
por_rango_r["nivel_retraso_predominante"] = np.nan
por_rango_r[_req_nulls] = np.nan
ep.drop(columns=["_rango_r"], inplace=True)

# Sección D: top 20 episodios con más requisiciones (tabla)
top_req = (
    ep[ep["num_requisiciones"] > 0]
    .nlargest(20, "num_requisiciones")
    [["episodio_id", "folio", "ext", "num_requisiciones",
      "monto_total_requisiciones", "dep_sol_principal",
      "tiempo_alta_admin_horas", "nivel_retraso"]]
    .copy()
)
top_req["tipo_fila"] = "top_episodios"
top_req["categoria"] = np.nan
top_req.rename(columns={
    "tiempo_alta_admin_horas": "tiempo_promedio_alta_admin_horas",
    "nivel_retraso":           "nivel_retraso_predominante",
}, inplace=True)
top_req[["conteo_episodios"] + _req_nulls] = np.nan

req_res = pd.concat([req_kpi, por_dep, por_rango_r, top_req], ignore_index=True)
req_res.to_csv(OUT / "requisiciones_resumen.csv", index=False)
print("OK requisiciones_resumen.csv")

# ── 11. monitor_operativo.csv ─────────────────────────────────────────────────
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

monitor.to_csv(OUT / "monitor_operativo.csv", index=False)
print("OK monitor_operativo.csv")

print(f"\nListo. 7 CSVs en {OUT.resolve()}")
