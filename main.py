import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CARREGAMENTO DA BASE

df = pd.read_csv(
    "dataset/Base Varejo.csv",
    sep=";"
)
print("Número de registros:", df.shape[0])
print("Número de colunas:", df.shape[1])

print("\nColunas:")
print(df.columns.tolist())

print("\nPrimeiras linhas:")
print(df.head())

# LIMPEZA INICIAL
# Remover colunas completamente vazias
df = df.dropna(axis=1, how="all")

# Remover espaços extras dos nomes das colunas
df.columns = df.columns.str.strip()

print(df.info())

# VERIFICAÇÃO DOS PROBLEMAS
# Valores nulos

print(df.isnull().sum())

# Duplicatas

print("Quantidade de duplicatas:", df.duplicated().sum())

# Categorias vazias

print("\n--- CATEGORIAS VAZIAS ---")
categorias_vazias = (
    df["PR_CAT"]
    .isna()
    .sum()
)
categorias_texto_vazio = (
    df["PR_CAT"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)
print("Categorias nulas:", categorias_vazias)
print("Categorias vazias:", categorias_texto_vazio)

# Datas inválidas

datas_convertidas = pd.to_datetime(
    df["DATA"],
    dayfirst=True,
    errors="coerce"
)
print(
    "Datas inválidas:",
    datas_convertidas.isna().sum()
)
# LIMPEZA DOS DADOS
# converter DATA

df["DATA"] = pd.to_datetime(
    df["DATA"],
    dayfirst=True,
    errors="coerce"
)
# Remover registros com data inválida
df = df.dropna(subset=["DATA"])

# tratar categorias
# Transformar valores vazios em NaN
df["PR_CAT"] = df["PR_CAT"].replace(
    ["", " "],
    np.nan
)
# Imputar categoria ausente usando a moda
if df["PR_CAT"].isna().sum() > 0:

    moda_categoria = df["PR_CAT"].mode()[0]

    df["PR_CAT"] = df["PR_CAT"].fillna(
        moda_categoria
    )

    print(
        "Categorias ausentes imputadas com a moda:",
        moda_categoria
    )
# eliminar duplicatas

duplicatas_antes = df.duplicated().sum()

df = df.drop_duplicates()

print(
    "Duplicatas removidas:",
    duplicatas_antes
)
print(
    "Registros após a limpeza:",
    len(df)
)
#INFORMAÇÕES APÓS A LIMPEZA

print(df.info())

print("\nValores nulos após limpeza:")
print(df.isnull().sum())

print("\nData inicial:", df["DATA"].min())
print("Data final:", df["DATA"].max())

# EXPLORAÇÃO DA BASE
# Clientes

print("\n--- CLIENTES ---")

print(
    "Clientes únicos:",
    df["CL_ID"].nunique()
)
# Produtos

print(
    "Produtos únicos:",
    df["PR_ID"].nunique()
)
# Categorias

print(
    df["PR_CAT"].value_counts()
)
# Gênero

print("\n--- GÊNERO ---")

print(
    df["CL_GENERO"].value_counts()
)
# Produtos mais frequentes

print(
    df["PR_NOME"]
    .value_counts()
    .head(10)
)
# ESTATÍSTICAS DESCRITIVAS

colunas_numericas = [
    "CO_ID",
    "CL_ID",
    "CL_EC",
    "CL_FHL",
    "PR_ID"
]
print(
    df[colunas_numericas].describe()
)
# Estatísticas individuais

for coluna in colunas_numericas:

    print("\n---", coluna, "---")

    print("Contagem:", df[coluna].count())
    print("Média:", df[coluna].mean())
    print("Mediana:", df[coluna].median())
    print("Desvio padrão:", df[coluna].std())
    print("Mínimo:", df[coluna].min())
    print("Máximo:", df[coluna].max())

    print(
        "Moda:",
        df[coluna].mode().tolist()
    )
    print(
        "Q1:",
        df[coluna].quantile(0.25)
    )

    print(
        "Q2:",
        df[coluna].quantile(0.50)
    )

    print(
        "Q3:",
        df[coluna].quantile(0.75)
    )
# ESTATÍSTICAS DOS CLIENTES

print("\n--- CL_EC ---")
print(
    df["CL_EC"]
    .value_counts()
    .sort_index()
)
print("\n--- CL_FHL ---")
print(
    df["CL_FHL"]
    .value_counts()
    .sort_index()
)

print("\n--- CL_SEG ---")
print(
    df["CL_SEG"]
    .value_counts()
)
# GROUPBY
# Compras por gênero

compras_genero = (
    df.groupby("CL_GENERO")
    .size()
    .sort_values(ascending=False)
)

print(compras_genero)

# Compras por segmento

compras_segmento = (
    df.groupby("CL_SEG")
    .size()
    .sort_values(ascending=False)
)

print(compras_segmento)

# Compras por categoria

compras_categoria = (
    df.groupby("PR_CAT")
    .size()
    .sort_values(ascending=False)
)

print(compras_categoria)

# Clientes únicos por segmento

print(
    df.groupby("CL_SEG")["CL_ID"]
    .nunique()
)
# Clientes únicos por gênero

print("\n--- CLIENTES ÚNICOS POR GÊNERO ---")

print(
    df.groupby("CL_GENERO")["CL_ID"]
    .nunique()
)
# GÊNERO X SEGMENTO

genero_segmento = pd.crosstab(
    df["CL_GENERO"],
    df["CL_SEG"]
)

print(genero_segmento)

# GÊNERO X CATEGORIA

genero_categoria = pd.crosstab(
    df["CL_GENERO"],
    df["PR_CAT"]
)

print(genero_categoria)

# PIVOT TABLE

pivot_genero_categoria = pd.pivot_table(
    df,
    index="CL_GENERO",
    columns="PR_CAT",
    values="PR_ID",
    aggfunc="count",
    fill_value=0
)
print(
    pivot_genero_categoria
)
# FREQUÊNCIA DE COMPRAS POR CLIENTE

compras_cliente = (
    df.groupby("CL_ID")
    .size()
)
print("\nEstatísticas:")
print(
    compras_cliente.describe()
)
print("\n10 clientes com mais registros:")

print(
    compras_cliente
    .sort_values(ascending=False)
    .head(10)
)
# ANÁLISE DO CO_ID

print(
    "CO_ID únicos:",
    df["CO_ID"].nunique()
)
print("\nRegistros por CO_ID:")

print(
    df.groupby("CO_ID")
    .size()
    .describe()
)
# Verificar se cada pedido pertence a
# um único cliente

print(
    df.groupby("CO_ID")["CL_ID"]
    .nunique()
    .describe()
)
# Verificar se cada pedido pertence a uma
# única data

print(
    df.groupby("CO_ID")["DATA"]
    .nunique()
    .describe()
)
# EVOLUÇÃO TEMPORAL

df["ANO"] = df["DATA"].dt.year

compras_ano = (
    df.groupby("ANO")
    .size()
)
print(compras_ano)

# GRÁFICOS

sns.set_theme(style="whitegrid")

# Gráfico 1: gênero

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="CL_GENERO",
    hue="CL_GENERO",
    palette="Set2",
    legend=False
)
plt.title("Quantidade de registros por gênero")
plt.xlabel("Gênero")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.show()

# Gráfico 2: categorias

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    y="PR_CAT",
    order=df["PR_CAT"].value_counts().index,
    hue="PR_CAT",
    palette="viridis",
    legend=False
)

plt.title("Quantidade de registros por categoria")
plt.xlabel("Quantidade")
plt.ylabel("Categoria")

plt.tight_layout()
plt.show()

# Gráfico 3: Top 10 produtos

top10_produtos = (
    df["PR_NOME"]
    .value_counts()
    .head(10)
)
plt.figure(figsize=(10, 6))

sns.barplot(
    x=top10_produtos.values,
    y=top10_produtos.index,
    hue=top10_produtos.index,
    palette="Blues_r",
    legend=False
)
plt.title("Top 10 produtos mais frequentes")
plt.xlabel("Quantidade de registros")
plt.ylabel("Produto")

plt.tight_layout()
plt.show()

# Gráfico 4: evolução anual

plt.figure(figsize=(9, 5))

sns.lineplot(
    x=compras_ano.index,
    y=compras_ano.values,
    marker="o"
)
plt.title("Evolução dos registros por ano")
plt.xlabel("Ano")
plt.ylabel("Quantidade de registros")

plt.tight_layout()
plt.show()

# CONCLUSÕES
print("CONCLUSÕES")

print("""
1. O gênero F apresentou maior quantidade de registros de compras
   do que o gênero M, indicando maior participação feminina na base.

2. A categoria ALIMENTOS apresentou a maior quantidade de registros,
   sendo a principal categoria de consumo analisada.

3. O segmento B concentrou a maior quantidade de registros entre
   os segmentos A, B e C.

4. PRESUNTO COZIDO foi o produto mais frequente da base,
   seguido por SARDINHA e GEL.

5. A base possui 1.000 clientes e milhares de pedidos, permitindo
   analisar diferentes padrões de comportamento de compra.

6. O período analisado vai de 2019 a 2022, possibilitando observar
   a evolução das compras ao longo do tempo.
""")





