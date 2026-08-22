import pandas as pd

df = pd.read_csv("articles.csv")
print(f"Filas: {len(df)}\n")
print("=== COLUMNAS ===")
print(list(df.columns))

print("\n=== CANDIDATAS A FILTRO ===")
for c in df.select_dtypes("object"):
    n = df[c].nunique()
    if n <= 60:                      # solo las manejables como enum
        print(f"\n--- {c}  ({n} valores)")
        print(df[c].value_counts().to_string())
    else:
        print(f"\n--- {c}  ({n} valores) -> demasiados, va a semántica")

print("\n=== TEXTO PARA EMBEBER ===")
print("detail_desc nulos:", df["detail_desc"].isna().sum())
print(df["detail_desc"].dropna().head(3).to_string())