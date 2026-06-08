import pandas as pd

base = "Data/raw/"

hospac = pd.read_csv(base + "Hospac.csv", nrows=3, low_memory=False)
print("=== HOSPAC columns ===")
print(hospac.columns.tolist())
print()
print("=== HOSPAC head ===")
print(hospac.to_string())
print()

hosffa = pd.read_csv(base + "Hosffa.csv", nrows=3, low_memory=False)
print("=== HOSFFA columns ===")
print(hosffa.columns.tolist())
print()

hosreq = pd.read_csv(base + "Hosreq.csv", nrows=3, low_memory=False)
print("=== HOSREQ columns ===")
print(hosreq.columns.tolist())
print()
print("=== HOSREQ head ===")
print(hosreq.to_string())
print()

hosfol = pd.read_csv(base + "Hosfol.csv", nrows=3, low_memory=False)
print("=== HOSFOL columns ===")
print(hosfol.columns.tolist())
print()

hosder = pd.read_csv(base + "Hosder.csv", nrows=3, low_memory=False)
print("=== HOSDER columns ===")
print(hosder.columns.tolist())
print()

# Hostransacciones no existe como CSV, buscar alternativa
import os
print("Archivos en Data/raw/:")
print(os.listdir("Data/raw/"))
