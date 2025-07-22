import pandas as pd

# Dati di esempio basati sulla tua tabella per Prato
# Qui gestiamo i punteggi per ogni indicatore e per ogni tool.
# Per "Baseline Water Stress", abbiamo 5 da WRI Aqueduct e 5 da WWF Water Risk Filter.
# Per "Water Depletion", abbiamo 3 da WRI Aqueduct e 2 da WWF Water Risk Filter.
# Il "Drought risk" è solo da WRI Aqueduct (3) e "Flood hazard" solo da WWF Water Risk Filter (2).

data = {
    'Tool': ['WRI Aqueduct', 'WRI Aqueduct', 'WWF Water Risk Filter', 'WWF Water Risk Filter', 'WRI Aqueduct', 'WWF Water Risk Filter'],
    'Indicator': ['Baseline Water Stress', 'Baseline Water Depletion', 'Water Depletion', 'Baseline water stress', 'Drought risk', 'Flood hazard'],
    'Score': [5, 3, 2, 5, 3, 2]
}
df = pd.DataFrame(data)

# --- Pre-elaborazione: Uniformare nomi indicatori e media dei punteggi per indicatore ---

# Uniformiamo i nomi degli indicatori che si riferiscono allo stesso concetto
df['Indicator_Unified'] = df['Indicator'].replace({
    'Baseline water stress': 'Baseline Water Stress',
    'Water Depletion': 'Baseline Water Depletion' # Assumiamo che siano concettualmente simili per la media
})

# Raggruppiamo per indicatore unificato e calcoliamo la media dei punteggi
# Questo è il passaggio chiave per "prendere i due risultati e fare la media"
df_avg_scores_per_indicator = df.groupby('Indicator_Unified')['Score'].mean().reset_index()

# --- Normalizzazione (Min-Max Scaling su scala 0-1) dei punteggi medi ---

# La scala originale dei punteggi è da 1 a 5.
min_score_originale = 1
max_score_originale = 5

# Applica la formula di Min-Max Scaling per portare ogni punteggio medio su una scala da 0 a 1
df_avg_scores_per_indicator['Score_Normalizzato_0_1'] = (
    df_avg_scores_per_indicator['Score'] - min_score_originale
) / (max_score_originale - min_score_originale)

# --- Calcolo della Media Semplice dei Punteggi Normalizzati (senza pesi) ---
# Ora prendiamo la media di questi punteggi normalizzati che sono già aggregati per indicatore.
score_finale_medio_normalizzato = df_avg_scores_per_indicator['Score_Normalizzato_0_1'].mean()

# --- Output dei Risultati ---
print("DataFrame con punteggi medi per indicatore e punteggi normalizzati:")
print(df_avg_scores_per_indicator)
print("\n--------------------------------------------------")
print(f"Score Finale Medio Normalizzato (scala 0-1): {score_finale_medio_normalizzato:.4f}")

# Per riportare su scala 1-5:
# Formula: Valore_1_5 = (Valore_0_1 * (Max_Desiderato - Min_Desiderato)) + Min_Desiderato
# Min_Desiderato = 1, Max_Desiderato = 5
score_finale_medio_normalizzato_su_scala_1_5 = (score_finale_medio_normalizzato * (5 - 1)) + 1

print(f"Score Finale Medio Normalizzato (riportato su scala 1-5): {score_finale_medio_normalizzato_su_scala_1_5:.2f}")

# --- Nota sul confronto con lo Score Finale originale (3.33) ---
# Il tuo score finale originale di 3.33 probabilmente derivava da una media più complessa
# o da una ponderazione implicita diversa da una semplice media di tutti gli indicatori.
# Questo nuovo calcolo fornisce una media "pulita" basata sui tuoi punteggi aggregati per indicatore.