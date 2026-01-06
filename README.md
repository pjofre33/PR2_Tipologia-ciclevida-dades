# Pràctica 2 - Anàlisi de Vinils de Jazz 2023

## Autors

- Pol Jofre i Campuzano
- Nil Masalles Domenech

## Descripció

Aquest projecte realitza una anàlisi completa del dataset de vinils de jazz venuts l'any 2023, obtingut de Discogs. L'objectiu principal és entendre quins factors influeixen en el preu dels vinils i si hi ha patrons identificables en el mercat.

## Estructura del projecte

```
PRACT2/
│
├── README.md                      
│
└── codi/
    ├── analisi_vinils.py              # Script principal amb tot l'anàlisi
    └── requirements.txt               # Dependències Python
```

## Fitxers necessaris

El projecte necessita accedir a:

- `../vinils_jazz_2023.csv` - Dataset original extret en la pràctica 1
## Com executar

## Entorn virtual

# Crear entorn virtual
python -m venv .venv

# Activar-lo
source .venv/bin/activate  # macOS/Linux

.venv\Scripts\activate     # Windows

# Instal·lar dependències
pip install -r codi/requirements.txt

# Executar
cd codi
python3 analisi_vinils.py
```

L'script generarà:

- **5 gràfics** al directori `codi/`:
  - `01_exploracio_inicial.png` - Visualització inicial del dataset
  - `02_despres_neteja.png` - Distribucions després de la neteja
  - `03_model_regressio.png` - Resultats dels models de regressió
  - `04_clustering.png` - Visualització dels clusters
  - `05_hipotesis.png` - Tests d'hipòtesi
- **1 dataset net** al directori arrel:
  - `/../vinils_jazz_2023_net.csv` - Dataset processat i net

## Contingut de l'anàlisi

L'script executa seqüencialment les següents seccions:

1. **Descripció del dataset** - Exploració inicial i estadístiques descriptives
2. **Integració i selecció** - Creació de variables derivades
3. **Neteja de dades** - Imputació de nuls, normalització de preus, gestió d'outliers
4. **Model de regressió** - Predicció del preu amb Linear Regression i Random Forest
5. **Clustering** - Identificació de grups amb K-Means
6. **Tests d'hipòtesi** - ANOVA i T-tests per validar diferències
7. **Conclusions** - Resum dels resultats i interpretació

## Preguntes de recerca

1. **Quins factors influeixen en el preu dels vinils de jazz?**
2. **Hi ha diferències de preu segons el país d'origen?**
3. **Les edicions limitades són més cares?**

## Resultats esperats

- Model de regressió amb R² i RMSE calculats
- Identificació de 3 clusters de vinils
- Tests estadístics per validar hipòtesis sobre preus
- Visualitzacions clares i interpretables
