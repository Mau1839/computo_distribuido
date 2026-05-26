import ray
import ray.data as rd
import pandas as pd

ray.init()

dfs = []

for year in range(2015, 2025):

    file = f"conjunto_de_datos/atus_anual_{year}.csv"

    print(f"Leyendo {file}...")

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False,
        dtype=str
    )

    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()

    # Quitar BOM si existe
    df.columns = df.columns.str.replace("ï»¿", "", regex=False)

    # Limpiar espacios en datos
    df["ID_ENTIDAD"] = df["ID_ENTIDAD"].str.strip()

    # Filtrar Chihuahua
    df = df[
        df["ID_ENTIDAD"].str.zfill(2) == "08"
    ]

    dfs.append(df)

# Unir todo
final_df = pd.concat(dfs, ignore_index=True)

# Crear dataset Ray
ds = rd.from_pandas(final_df)

print(ds.take(5))

print("Total registros:", ds.count())

# Guardar CSV correcto
final_df.to_csv(
    "chihuahua_2015_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Archivo generado correctamente")