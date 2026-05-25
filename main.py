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
        low_memory=False
    )

    df = df[
        df["ID_ENTIDAD"].astype(str).str.zfill(2) == "08"
    ]

    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

ds = rd.from_pandas(final_df)

print(ds.take(5))

print("Total registros:", ds.count())

final_df.to_csv(
    "chihuahua_2015_2024.csv",
    index=False
)

print("Archivo generado correctamente")