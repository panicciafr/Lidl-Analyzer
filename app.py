import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE APP ---
st.set_page_config(page_title="Lidl Receipt Analyzer", layout="wide")
st.title("🛒 App Gestione e Analisi Scontrini Lidl")

# --- CACHE DEL MODELLO AI ---
@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- REGEX MIGLIORATE ---
IMPORTO = r"-?\s?\d{1,5}[.,]\d{2}"
RE_DATA_ORA = re.compile(r"(\d{2})[-/.](\d{2})[-/.](\d{4})\s+(\d{1,2})[:.](\d{2})")
RE_TOTALE = re.compile(r"TOTALE\s+COMPLESSIVO\s+(" + IMPORTO + r")", re.IGNORECASE)
RE_PRODOTTO = re.compile(r"^(?P<nome>.+?)\s+(?P<iva>\d{1,2})\s*[%96°a-zA-Z]*\s+(?P<prezzo>" + IMPORTO + r")\s*$")
RE_DETTAGLIO_QTA = re.compile(r"^Cad\.?\s+(?P<unitario>\d{1,5}[.,]\d{2})\s+Pz\.?\s*(?P<qta>\d+)", re.IGNORECASE)
RE_FINE_PRODOTTI = re.compile(r"VALORE\s+SCONTI|SUB\s*TOTALE|TOTALE\s+COMPLESSIVO", re.IGNORECASE)

def normalizza_importo(testo: str) -> str:
    return testo.replace(" ", "").replace(",", ".").replace("−", "-").replace("–", "-").replace("—", "-")

def elabora_immagine(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return pytesseract.image_to_string(img, lang="ita", config="--psm 6")

def estrai_dati(testo):
    righe = [r.replace("−", "-").strip() for r in testo.splitlines() if r.strip()]
    data_ora, totale = "", ""
    
    m_data = RE_DATA_ORA.search(testo)
    if m_data:
        g, mese, anno, ore, minuti = m_data.groups()
        data_ora = f"{g}/{mese}/{anno} {int(ore):02d}:{minuti}"

    m_tot = RE_TOTALE.search(testo)
    if m_tot:
        totale = normalizza_importo(m_tot.group(1))

    fine = len(righe)
    for i, riga in enumerate(righe):
        if RE_FINE_PRODOTTI.search(riga):
            fine = i
            break

    prodotti = []
    for riga in righe[:fine]:
        m_prod = RE_PRODOTTO.match(riga)
        if m_prod:
            prezzo = normalizza_importo(m_prod.group("prezzo"))
            prodotti.append({
                "Prodotto": m_prod.group("nome").strip(),
                "Prezzo_Acquisto": prezzo,
                "Quantita": "1",
                "Prezzo_Unitario": prezzo
            })
            continue
            
        m_qta = RE_DETTAGLIO_QTA.match(riga)
        if m_qta and prodotti:
            prodotti[-1]["Quantita"] = m_qta.group("qta")
            prodotti[-1]["Prezzo_Unitario"] = normalizza_importo(m_qta.group("unitario"))

    return data_ora, totale, prodotti

# --- INTERFACCIA STREAMLIT ---
tab1, tab2 = st.tabs(["📸 1. Aggiorna Database (OCR)", "📊 2. Analisi Dati & AI"])

# ==========================================
# TAB 1: ESTRAZIONE OCR & MERGING
# ==========================================
with tab1:
    st.header("Carica nuovi scontrini")
    
    # Inizializziamo una "chiave" nella memoria di sessione per il reset
    if "reset_key" not in st.session_state:
        st.session_state.reset_key = 0
        
    # TASTO SCIACQUONE
    if st.button("🚽 Svuota tutti i file (Sciacquone)"):
        st.session_state.reset_key += 1 # Cambiando la chiave, l'uploader si resetta
        st.rerun() # Forza l'aggiornamento istantaneo dell'interfaccia
    
    # Aggiungiamo la chiave dinamica al file_uploader
    uploaded_images = st.file_uploader(
        "Carica gli scontrini in formato immagine (.png, .jpg)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key=f"uploader_immagini_{st.session_state.reset_key}" # Chiave dinamica
    )
    
    uploaded_db = st.file_uploader(
        "Se ne hai già uno, carica il database esistente per accodare i dati (.csv)", 
        type=['csv'],
        key=f"uploader_db_{st.session_state.reset_key}" # Chiave dinamica
    )
    
    if st.button("Avvia Estrazione OCR"):
        if not uploaded_images:
            st.warning("Carica almeno un'immagine per procedere.")
        else:
            with st.spinner("Estrazione testo in corso..."):
                existing_df = pd.DataFrame()
                if uploaded_db:
                    existing_df = pd.read_csv(uploaded_db, sep=';', encoding='utf-8-sig')
                
                new_data = []
                for img in uploaded_images:
                    testo_ocr = elabora_immagine(img.read())
                    data_ora, totale, prodotti = estrai_dati(testo_ocr)
                    for p in prodotti:
                        new_data.append({
                            "File_Scontrino": img.name,
                            "Data_Ora": data_ora,
                            "Prodotto": p["Prodotto"],
                            "Quantita": p["Quantita"],
                            "Prezzo_Unitario": p["Prezzo_Unitario"],
                            "Prezzo_Totale_Riga": p["Prezzo_Acquisto"],
                            "Importo_Totale_Scontrino": totale
                        })
                
                new_df = pd.DataFrame(new_data)
                
                if not existing_df.empty and not new_df.empty:
                    scontrini_esistenti = existing_df[['Data_Ora', 'Importo_Totale_Scontrino']].drop_duplicates()
                    merged_check = new_df.merge(scontrini_esistenti, on=['Data_Ora', 'Importo_Totale_Scontrino'], how='left', indicator=True)
                    new_df_filtrato = merged_check[merged_check['_merge'] == 'left_only'].drop(columns=['_merge'])
                    final_df = pd.concat([existing_df, new_df_filtrato], ignore_index=True)
                    righe_aggiunte = len(new_df_filtrato)
                elif not new_df.empty:
                    final_df = new_df
                    righe_aggiunte = len(new_df)
                else:
                    final_df = existing_df
                    righe_aggiunte = 0

                if not final_df.empty:
                    st.success(f"Completato! Aggiunte {righe_aggiunte} nuove righe valide. Somme preservate.")
                    st.dataframe(final_df.tail(10))
                    
                    csv = final_df.to_csv(sep=';', index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        label="⬇️ Scarica Database Aggiornato (.csv)",
                        data=csv,
                        file_name='scontrini_database_aggiornato.csv',
                        mime='text/csv',
                    )
                else:
                    st.error("Nessun dato valido estrapolato dalle immagini.")

# ==========================================
# TAB 2: ANALISI STATISTICA
# ==========================================
with tab2:
    st.header("Dashboard Statistica")
    analysis_db = st.file_uploader("Carica il tuo database aggiornato (.csv) per avviare l'analisi", type=['csv'], key="analysis_uploader")
    
    if analysis_db:
        if st.button("Genera Dashboard"):
            with st.spinner("Classificazione AI e calcoli statistici in corso..."):
                df = pd.read_csv(analysis_db, sep=';', encoding='utf-8-sig')
                
                df['Data_Ora'] = pd.to_datetime(df['Data_Ora'], format='%d/%m/%Y %H:%M')
                for col in ['Quantita', 'Prezzo_Unitario', 'Prezzo_Totale_Riga', 'Importo_Totale_Scontrino']:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

                esclusioni = ['Coupon Lidl Points', 'SACCH.ORTOFR.BIOGRAD', 'Offerta Lidl Plus']
                pattern = '|'.join(esclusioni)
                df = df[~df['Prodotto'].str.contains(pattern, case=False, na=False)]

                # FIX AI: Rimuoviamo i valori nulli e forziamo la colonna Prodotto a essere puro Testo
                df = df.dropna(subset=['Prodotto'])
                df['Prodotto'] = df['Prodotto'].astype(str).str.strip()

                nlp_model = load_nlp_model()
                # Assicuriamoci di passare al modello una vera e propria lista (tolist) di stringhe
                prodotti_unici = df['Prodotto'].unique().tolist()
                embeddings = nlp_model.encode(prodotti_unici)

                clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='ward')
                clustering_model.fit(embeddings)

                mappa_cluster = {prod: f"AI_Category_{cluster_id}" for prod, cluster_id in zip(prodotti_unici, clustering_model.labels_)}
                df['Categoria_Semantica'] = df['Prodotto'].map(mappa_cluster)
                nomi_categorie = df.groupby('Categoria_Semantica')['Prodotto'].agg(lambda x: x.value_counts().index[0]).to_dict()
                df['Categoria_Semantica'] = df['Categoria_Semantica'].map(nomi_categorie)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📦 Top 10 per Volume (Unità)")
                    top10_qty = df.groupby('Categoria_Semantica')['Quantita'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(top10_qty.rename("Unità Acquistate"), use_container_width=True)

                    st.subheader("💰 Top 10 per Spesa Totale Assoluta (€)")
                    top10_spend = df.groupby('Categoria_Semantica')['Prezzo_Totale_Riga'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(top10_spend.rename("Spesa (€)"), use_container_width=True)

                with col2:
                    st.subheader("🛒 Top 10 per Frequenza nel Carrello")
                    top10_freq = df.groupby('Categoria_Semantica')['File_Scontrino'].nunique().sort_values(ascending=False).head(10)
                    st.dataframe(top10_freq.rename("N. Scontrini"), use_container_width=True)

                    st.subheader("📈 Top 10 Record Prezzo Unitario (€)")
                    top10_prices = df.groupby('Categoria_Semantica')['Prezzo_Unitario'].max().sort_values(ascending=False).head(10)
                    st.dataframe(top10_prices.rename("Prezzo Max (€)"), use_container_width=True)

                st.divider()
                st.subheader("🔗 Regole di Associazione (Acquistati Spesso Insieme)")
                basket = (df.groupby(['File_Scontrino', 'Categoria_Semantica'])['Quantita']
                          .sum().unstack().reset_index().fillna(0).set_index('File_Scontrino'))
                basket_sets = basket.map(lambda x: True if x > 0 else False)

                frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    if not rules.empty:
                        rules = rules.sort_values('confidence', ascending=False).head(5)
                        for _, row in rules.iterrows():
                            antecedents = ", ".join(list(row['antecedents']))
                            consequents = ", ".join(list(row['consequents']))
                            st.write(f"- Chi acquista **[{antecedents}]** ha il **{row['confidence']*100:.1f}%** di probabilità di prendere anche **[{consequents}]**")
                    else:
                        st.info("Nessuna associazione forte rilevata con i volumi attuali.")
                else:
                    st.info("Dati insufficienti per generare regole di associazione.")

                st.divider()
                st.subheader("📊 Visualizzazione Distribuzioni")
                sns.set_theme(style="whitegrid")
                fig, axes = plt.subplots(3, 1, figsize=(12, 18))

                sns.barplot(x=top10_spend.values, y=top10_spend.index, ax=axes[0], hue=top10_spend.index, palette="viridis", legend=False)
                axes[0].set_title('Top 10 Prodotti per Spesa Assoluta (€)', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Spesa Totale (€)')
                axes[0].set_ylabel('')

                df_top_freq = df[df['Categoria_Semantica'].isin(top10_freq.index)]
                sns.boxplot(x='Prezzo_Unitario', y='Categoria_Semantica', data=df_top_freq, ax=axes[1], hue='Categoria_Semantica', palette="Set2", legend=False)
                axes[1].set_title('Distribuzione Prezzi Unitari (10 Prodotti Più Acquistati)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Prezzo Unitario (€)')
                axes[1].set_ylabel('')

                totali_unici = df.drop_duplicates(subset=['File_Scontrino'])['Importo_Totale_Scontrino']
                sns.boxplot(x=totali_unici, ax=axes[2], color="skyblue")
                axes[2].set_title('Distribuzione degli Importi Totali Scontrino (€)', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Importo Totale (€)')

                plt.tight_layout()
                st.pyplot(fig)
