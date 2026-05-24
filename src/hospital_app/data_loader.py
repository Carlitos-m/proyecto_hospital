import pandas as pd
import zipfile
from io import BytesIO
from pathlib import Path
nuevas_llaves = ["Hosder","Hosffa","Hosfol","Hoshab","Hospac","Hosreq","Hostha","Hostransacciones"]
def cargar_dfs(ruta_zip=None):
    if ruta_zip is None:
        # Si no se da ruta, busca en data/raw/ dentro del proyecto
        ruta_zip = Path(__file__).resolve().parents[2] / "data" / "raw" / "tablas_assist_parquet.zip"
    else:
        ruta_zip = Path(ruta_zip)

    bases_de_interes = [
        "HOSPAC.parquet", "HOSTRANSACCIONES.parquet", "HOSREQ.parquet",
        "HOSDER.parquet", "HOSFOL.parquet", "HOSFFA.parquet",
        "HOSHAB.parquet", "HOSTHA.parquet"
    ]
    nuevas_llaves = ["Hosder","Hosffa","Hosfol","Hoshab","Hospac","Hosreq","Hostha","Hostransacciones"]

    dfs = {}
    with zipfile.ZipFile(ruta_zip, 'r') as z:
        parquet_files = [f for f in z.namelist() if f.endswith(".parquet") and any(base in f for base in bases_de_interes)]
        for file in parquet_files:
            with z.open(file) as f:
                dfs[file] = pd.read_parquet(BytesIO(f.read()), engine="pyarrow")

    dfs = {nueva: dfs[vieja] for vieja, nueva in zip(parquet_files, nuevas_llaves)}
    return dfs


dfs= cargar_dfs(r"C:\Users\carlo\OneDrive\Documentos\Carlos_universidad\oct_semestre\proyectosMatematicas\tablas_assist_parquet.zip")
#%%

#%%
