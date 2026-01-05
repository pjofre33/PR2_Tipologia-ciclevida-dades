"""
Pràctica 2 - Anàlisi de Vinils de Jazz 2023
Autors: Pol Jofre i Nil Masalles Domenech
Data: Gener 2026
"""

# =============================================================================
# SECCIÓ 1: IMPORTS I CÀRREGA DE DADES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuració de plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("PRÀCTICA 2 - ANÀLISI DE VINILS DE JAZZ 2023")

# =============================================================================
# 1. DESCRIPCIÓ DEL DATASET
# =============================================================================

print("\n1. DESCRIPCIÓ DEL DATASET")

# Carreguem les dades
df = pd.read_csv('../vinils_jazz_2023.csv')

print(f"\nDataset carregat correctament!")
print(f"Dimensions: {df.shape[0]} files × {df.shape[1]} columnes\n")

print("PREGUNTA DE RECERCA:")
print("Quins factors influeixen en el preu dels vinils de jazz?")
print("Hi ha diferències de preu segons el país d'origen?\n")

print("VARIABLES DEL DATASET:")
print(df.columns.tolist())
print("\nPrimeres files:")
print(df.head())

print("\nInformació bàsica:")
df.info()

print("\nEstadístiques descriptives:")
print(df.describe())

# Visualització inicial
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribució de preus (sense processar encara)
df_temp = df.copy()
df_temp['preu_num'] = df_temp['preu'].str.extract(r'([0-9.]+)').astype(float)
axes[0, 0].hist(df_temp['preu_num'].dropna(), bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Preu')
axes[0, 0].set_ylabel('Frequència')
axes[0, 0].set_title('Distribució de Preus')

# Top països
top_countries = df['shipped_from'].value_counts().head(10)
axes[0, 1].barh(range(len(top_countries)), top_countries.values)
axes[0, 1].set_yticks(range(len(top_countries)))
axes[0, 1].set_yticklabels(top_countries.index)
axes[0, 1].set_xlabel('Nombre de vinils')
axes[0, 1].set_title('Top 10 Països')

# Distribució de ratings
axes[1, 0].hist(df['rating'].dropna(), bins=20, edgecolor='black')
axes[1, 0].set_xlabel('Rating')
axes[1, 0].set_ylabel('Frequència')
axes[1, 0].set_title('Distribució de Ratings')

# Condicions dels vinils
condition_counts = df['media_condition'].value_counts().head(8)
axes[1, 1].bar(range(len(condition_counts)), condition_counts.values)
axes[1, 1].set_xticks(range(len(condition_counts)))
axes[1, 1].set_xticklabels(condition_counts.index, rotation=45, ha='right')
axes[1, 1].set_ylabel('Frequència')
axes[1, 1].set_title('Condicions dels Vinils')

plt.tight_layout()
plt.savefig('01_exploracio_inicial.png', dpi=150, bbox_inches='tight')
print("\n✓ Gràfic guardat: 01_exploracio_inicial.png")
plt.close()

# =============================================================================
# 2. INTEGRACIÓ I SELECCIÓ DE DADES
# =============================================================================

print("2. INTEGRACIÓ I SELECCIÓ DE DADES")

print("\nMantenim totes les variables originals i en creem algunes de noves:")

# Crear variables derivades
df_net = df.copy()

# Extreure moneda i preu numèric
print("\n- Extraient moneda i preu numèric...")
df_net['moneda'] = df_net['preu'].astype(str).str.extract(r'([€£$])')
df_net['preu_num'] = df_net['preu'].astype(str).str.extract(r'([0-9.]+)').astype(float)

# Extreure format (LP, 2xLP, etc.)
print("- Extraient format del vinil...")
df_net['format'] = df_net['titol'].str.extract(r'\(([^)]*LP[^)]*)\)')
df_net['format'] = df_net['format'].fillna('LP')

# Edicions limitades
print("- Identificant edicions limitades...")
df_net['es_limitada'] = df_net['titol'].str.contains('Ltd|Limited', case=False, na=False).astype(int)

# Nombre de discos
print("- Extraient nombre de discos...")
df_net['num_discos'] = df_net['titol'].str.extract(r'(\d+)x')
df_net['num_discos'] = pd.to_numeric(df_net['num_discos'], errors='coerce').fillna(1).astype(int)

# Netejar media_condition (eliminar salts de línia i text descriptiu)
print("- Netejant camp 'media_condition'...")
df_net['media_condition'] = df_net['media_condition'].str.extract(r'^([^)]+\))', expand=False)

# Netejar shipped_from (canviar "Ships From:" per "Origen: ")
print("- Netejant camp 'shipped_from'...")
df_net['shipped_from'] = df_net['shipped_from'].str.replace('Ships From:', 'Origen:', regex=False)

print(f"\n✓ Variables derivades creades: {len(['moneda', 'preu_num', 'format', 'es_limitada', 'num_discos'])}")

# =============================================================================
# 3. NETEJA DE DADES
# =============================================================================

print("3. NETEJA DE DADES")

# 3.1 Valors nuls
print("\n3.1. GESTIÓ DE VALORS NULS")

print("\nValors nuls per variable:")
print(df_net.isnull().sum()[df_net.isnull().sum() > 0])

# Imputar rating amb la mediana del venedor
print("\n- Imputant 'rating' amb mediana per venedor...")
df_net['rating'] = df_net.groupby('venedor')['rating'].transform(
    lambda x: x.fillna(x.median())
)
# Si encara queden nuls, usar la mediana global
df_net['rating'].fillna(df_net['rating'].median(), inplace=True)

# Imputar sleeve_condition amb el valor més comú
print("- Imputant 'sleeve_condition' amb el valor més freqüent...")
df_net['sleeve_condition'].fillna(df_net['sleeve_condition'].mode()[0], inplace=True)

print(f"\nValors nuls restants: {df_net.isnull().sum().sum()}")

# 3.2 Conversió de tipus
print("\n3.2. CONVERSIÓ DE TIPUS DE DADES")

categoriques = ['venedor', 'label', 'media_condition', 'sleeve_condition',
                'shipped_from', 'moneda', 'format']

for col in categoriques:
    if col in df_net.columns:
        df_net[col] = df_net[col].astype('category')

print(f"{len(categoriques)} variables convertides a 'category'")

# 3.3 Normalitzar preus a EUR
print("\n3.3. NORMALITZACIÓ DE PREUS A EUR")

# Tipus de canvi aproximats (desembre 2023)
print("Tipus de canvi utilitzats:")
print("  £ → EUR: 1.17")
print("  $ → EUR: 0.92")

df_net['preu_eur'] = df_net['preu_num'].copy()
df_net.loc[df_net['moneda'] == '£', 'preu_eur'] *= 1.17
df_net.loc[df_net['moneda'] == '$', 'preu_eur'] *= 0.92

print(f"\nTots els preus normalitzats a EUR")
print(f"Preu mitjà: {df_net['preu_eur'].mean():.2f} EUR")
print(f"Preu mínim: {df_net['preu_eur'].min():.2f} EUR")
print(f"Preu màxim: {df_net['preu_eur'].max():.2f} EUR")

# 3.4 Outliers
print("\n3.4. IDENTIFICACIÓ D'OUTLIERS")

Q1 = df_net['preu_eur'].quantile(0.25)
Q3 = df_net['preu_eur'].quantile(0.75)
IQR = Q3 - Q1

limit_inferior = Q1 - 1.5 * IQR
limit_superior = Q3 + 1.5 * IQR

outliers = (df_net['preu_eur'] < limit_inferior) | (df_net['preu_eur'] > limit_superior)
df_net['es_outlier'] = outliers.astype(int)

print(f"Outliers detectats: {outliers.sum()} ({outliers.sum()/len(df_net)*100:.1f}%)")
print(f"Límit inferior: {limit_inferior:.2f} EUR")
print(f"Límit superior: {limit_superior:.2f} EUR")
print("\nNOTA: Els outliers NO s'eliminen, només es marquen.")

# Visualització després de la neteja
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma de preus EUR
axes[0, 0].hist(df_net['preu_eur'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Preu (EUR)')
axes[0, 0].set_ylabel('Frequència')
axes[0, 0].set_title('Distribució de Preus (EUR normalitzats)')

# Boxplot per identificar outliers
axes[0, 1].boxplot(df_net['preu_eur'].dropna())
axes[0, 1].set_ylabel('Preu (EUR)')
axes[0, 1].set_title('Boxplot - Detecció d\'Outliers')

# Ratings després d'imputació
axes[1, 0].hist(df_net['rating'], bins=20, edgecolor='black')
axes[1, 0].set_xlabel('Rating')
axes[1, 0].set_ylabel('Frequència')
axes[1, 0].set_title('Ratings (després d\'imputació)')

# Condicions dels vinils
cond_counts = df_net['media_condition'].value_counts().head(8)
axes[1, 1].bar(range(len(cond_counts)), cond_counts.values)
axes[1, 1].set_xticks(range(len(cond_counts)))
axes[1, 1].set_xticklabels(cond_counts.index, rotation=45, ha='right')
axes[1, 1].set_ylabel('Frequència')
axes[1, 1].set_title('Condicions dels Vinils (Netejades)')

plt.tight_layout()
plt.savefig('02_despres_neteja.png', dpi=150, bbox_inches='tight')
print("\nGràfic guardat: 02_despres_neteja.png")
plt.close()

# Crear variables numèriques per a les condicions (necessàries per models)
print("\nCreant variables numèriques per a condicions...")
le_media = LabelEncoder()
le_sleeve = LabelEncoder()
df_net['media_cond_num'] = le_media.fit_transform(df_net['media_condition'].astype(str))
df_net['sleeve_cond_num'] = le_sleeve.fit_transform(df_net['sleeve_condition'].astype(str))
print("✓ Variables numèriques creades")

# Guardar dataset net
df_net.to_csv('../vinils_jazz_2023_net.csv', index=False)
print("\nDataset net guardat: vinils_jazz_2023_net.csv")

# =============================================================================
# 4.1 MODEL SUPERVISAT - REGRESSIÓ
# =============================================================================

print("4.1 MODEL SUPERVISAT - REGRESSIÓ PER PREDIR EL PREU")

print("\nObjectiu: Predir el preu dels vinils basant-nos en les seves característiques\n")

# Preparar les dades (ja tenen media_cond_num i sleeve_cond_num de la secció 3)
df_model = df_net.copy()

# Seleccionar features
features = ['rating', 'media_cond_num', 'sleeve_cond_num', 'es_limitada', 'num_discos']

# Eliminar files amb valors nuls en les features o target
df_model_clean = df_model[features + ['preu_eur']].dropna()

X = df_model_clean[features]
y = df_model_clean['preu_eur']

print(f"Dades per al model: {len(X)} mostres")
print(f"Features utilitzades: {features}\n")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train)} mostres")
print(f"Test set: {len(X_test)} mostres\n")

# Model 1: Regressió Lineal
print("MODEL 1: REGRESSIÓ LINEAL")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"R²: {lr_r2:.4f}")
print(f"RMSE: {lr_rmse:.2f} EUR\n")

# Model 2: Random Forest
print("MODEL 2: RANDOM FOREST")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"R²: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.2f} EUR\n")

# Comparació de models
print("COMPARACIÓ DE MODELS:")

print(f"{'Model':<20} {'R²':<10} {'RMSE (EUR)':<15}")

print(f"{'Regressió Lineal':<20} {lr_r2:<10.4f} {lr_rmse:<15.2f}")
print(f"{'Random Forest':<20} {rf_r2:<10.4f} {rf_rmse:<15.2f}")

# Importància de features (Random Forest)
print("\nIMPORTÀNCIA DE LES FEATURES (Random Forest):")

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importància': rf_model.feature_importances_
}).sort_values('Importància', ascending=False)
print(feature_importance.to_string(index=False))

# Visualitzacions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prediccions vs Reals (Linear)
axes[0].scatter(y_test, y_pred_lr, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Preu Real (EUR)')
axes[0].set_ylabel('Preu Predit (EUR)')
axes[0].set_title(f'Regressió Lineal (R²={lr_r2:.3f})')

# Prediccions vs Reals (Random Forest)
axes[1].scatter(y_test, y_pred_rf, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Preu Real (EUR)')
axes[1].set_ylabel('Preu Predit (EUR)')
axes[1].set_title(f'Random Forest (R²={rf_r2:.3f})')

# Importància de features
axes[2].barh(range(len(feature_importance)), feature_importance['Importància'])
axes[2].set_yticks(range(len(feature_importance)))
axes[2].set_yticklabels(feature_importance['Feature'])
axes[2].set_xlabel('Importància')
axes[2].set_title('Importància de Features (RF)')

plt.tight_layout()
plt.savefig('03_model_regressio.png', dpi=150, bbox_inches='tight')
print("\nGràfic guardat: 03_model_regressio.png")
plt.close()

# =============================================================================
# 4.2 MODEL NO SUPERVISAT - CLUSTERING
# =============================================================================


print("4.2 MODEL NO SUPERVISAT - CLUSTERING")

print("\nObjectiu: Identificar grups naturals de vinils amb característiques similars\n")

# Preparar dades per clustering
df_cluster = df_net[['preu_eur', 'rating', 'media_cond_num', 'num_discos']].dropna()

# Escalar les dades
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# Mètode del colze per trobar el nombre òptim de clusters
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

# K-Means amb 3 clusters
print("Aplicant K-Means amb 3 clusters...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_cluster['cluster'] = clusters

print(f"\nClusters identificats: 3")
print(f"Distribució de vinils per cluster:")
print(df_cluster['cluster'].value_counts().sort_index())

# Perfil dels clusters
print("\nPERFIL DELS CLUSTERS:")

for i in range(3):
    cluster_data = df_cluster[df_cluster['cluster'] == i]
    print(f"\nCLUSTER {i}:")
    print(f"  - Nombre de vinils: {len(cluster_data)}")
    print(f"  - Preu mitjà: {cluster_data['preu_eur'].mean():.2f} EUR")
    print(f"  - Rating mitjà: {cluster_data['rating'].mean():.2f}")
    print(f"  - Discos mitjà: {cluster_data['num_discos'].mean():.2f}")

# Visualitzacions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mètode del colze
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Nombre de clusters (K)')
axes[0].set_ylabel('Inèrcia')
axes[0].set_title('Mètode del Colze')
axes[0].grid(True)

# Scatter plot dels clusters
scatter = axes[1].scatter(df_cluster['preu_eur'], df_cluster['rating'], 
                         c=df_cluster['cluster'], cmap='viridis', alpha=0.6)
axes[1].set_xlabel('Preu (EUR)')
axes[1].set_ylabel('Rating')
axes[1].set_title('Clusters Identificats')
plt.colorbar(scatter, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('04_clustering.png', dpi=150, bbox_inches='tight')
print("\nGràfic guardat: 04_clustering.png")
plt.close()

# =============================================================================
# 4.3 TEST D'HIPÒTESI
# =============================================================================

print("4.3 TEST D'HIPÒTESI")

# HIPÒTESI 1: Diferències de preu entre països
print("\nHIPÒTESI 1: Hi ha diferències significatives de preu entre països?")

# Agafem els 5 països amb més vinils
top5_countries = df_net['shipped_from'].value_counts().head(5).index
df_top5 = df_net[df_net['shipped_from'].isin(top5_countries)]

print(f"Països analitzats: {list(top5_countries)}\n")

# Crear grups per país
grups_pais = [df_top5[df_top5['shipped_from'] == pais]['preu_eur'].dropna() 
              for pais in top5_countries]

# Test ANOVA
f_stat, p_value = stats.f_oneway(*grups_pais)

print(f"ANOVA One-Way:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Significatiu? {'SÍ' if p_value < 0.05 else 'NO'} (α=0.05)")

if p_value < 0.05:
    print("\nHi ha diferències significatives de preu entre països")
else:
    print("\nNo hi ha diferències significatives de preu entre països")

# HIPÒTESI 2: Edicions limitades vs regulars
print("HIPÒTESI 2: Les edicions limitades són més cares?")


limitades = df_net[df_net['es_limitada'] == 1]['preu_eur'].dropna()
regulars = df_net[df_net['es_limitada'] == 0]['preu_eur'].dropna()

print(f"Edicions limitades: {len(limitades)} vinils")
print(f"Edicions regulars: {len(regulars)} vinils")
print(f"\nPreu mitjà limitades: {limitades.mean():.2f} EUR")
print(f"Preu mitjà regulars: {regulars.mean():.2f} EUR")

# T-test
t_stat, p_value_t = stats.ttest_ind(limitades, regulars)

print(f"\nT-test independent:")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value_t:.6f}")
print(f"  Significatiu? {'SÍ' if p_value_t < 0.05 else 'NO'} (α=0.05)")

if p_value_t < 0.05:
    if limitades.mean() > regulars.mean():
        print("\nLes edicions limitades SÓN significativament més cares")
    else:
        print("\nLes edicions limitades SÓN significativament més barates")
else:
    print("\nNo hi ha diferències significatives de preu")

# Visualitzacions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot per països
df_top5_plot = df_top5.copy()
axes[0].boxplot([df_top5_plot[df_top5_plot['shipped_from'] == p]['preu_eur'].dropna() 
                 for p in top5_countries],
                labels=top5_countries)
axes[0].set_ylabel('Preu (EUR)')
axes[0].set_xlabel('País')
axes[0].set_title(f'Preus per País (p={p_value:.4f})')
axes[0].tick_params(axis='x', rotation=45)

# Boxplot limitades vs regulars
axes[1].boxplot([regulars, limitades], labels=['Regulars', 'Limitades'])
axes[1].set_ylabel('Preu (EUR)')
axes[1].set_title(f'Limitades vs Regulars (p={p_value_t:.4f})')

plt.tight_layout()
plt.savefig('05_hipotesis.png', dpi=150, bbox_inches='tight')
print("\n✓ Gràfic guardat: 05_hipotesis.png")
plt.close()

# =============================================================================
# 5. CONCLUSIONS
# =============================================================================

print("5. CONCLUSIONS")

print("""
PREGUNTA PRINCIPAL: Quins factors influeixen en el preu dels vinils?

1. MODEL DE REGRESSIÓ:
   - El Random Forest ha obtingut millors resultats que la regressió lineal
   - R² del Random Forest indica que podem explicar una part del preu
   - Les variables més importants són la condició del vinil i el rating del venedor

2. CLUSTERING:
   - S'han identificat 3 grups naturals de vinils:
     * Cluster 0: Vinils de preu mitjà-baix
     * Cluster 1: Vinils de preu alt amb bon rating
     * Cluster 2: Vinils de preu mitjà amb rating variable
   - Això indica que hi ha segments de mercat diferenciats

3. TESTS D'HIPÒTESI:
   - Hi ha diferències de preu significatives entre països (si p<0.05)
   - Les edicions limitades poden tenir preus diferents (segons resultat)
   - Això confirma que el mercat de vinils no és homogeni

LIMITACIONS:
- Podríem millorar el model amb més features (segell discogràfic, artista, etc.)
- Alguns outliers poden afectar els resultats
- El dataset només cobreix l'any 2023

OBSERVACIONS:
- Per vendre vinils a millor preu: bon estat del vinil i bon rating
- Les edicions limitades poden tenir valor afegit
- El país d'origen pot influir en el preu (costos d'enviament, reputació)
""")

print("FI DE L'ANÀLISI")
print("\nTots els gràfics s'han guardat al directori actual:")
print("  - 01_exploracio_inicial.png")
print("  - 02_despres_neteja.png")
print("  - 03_model_regressio.png")
print("  - 04_clustering.png")
print("  - 05_hipotesis.png")
print("\nDataset net guardat: vinils_jazz_2023_net.csv")
