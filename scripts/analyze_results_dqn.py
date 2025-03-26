# analyze_results_dqn.py
# Lee el último archivo de resultados y grafica tubos y recompensas

import os
import pandas as pd
import matplotlib.pyplot as plt

# Buscar el archivo CSV más reciente
def get_latest_csv(prefix="flappy_test_results_"):
    csv_files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("❌ No se encontraron archivos CSV de resultados.")
    latest = max(csv_files, key=os.path.getctime)
    print(f"📄 Archivo cargado: {latest}")
    return latest

# Leer datos
df = pd.read_csv(get_latest_csv())

# Cálculo de estadísticas
max_score = df["Tubos Pasados"].max()
min_score = df["Tubos Pasados"].min()
avg_score = df["Tubos Pasados"].mean()

max_reward = df["Recompensa Total"].max()
min_reward = df["Recompensa Total"].min()
avg_reward = df["Recompensa Total"].mean()

print("\n📊 Estadísticas de los resultados:")
print(f"   🎯 Tubos: max={max_score}, min={min_score}, avg={avg_score:.2f}")
print(f"   💰 Recompensa: max={max_reward:.2f}, min={min_reward:.2f}, avg={avg_reward:.2f}")

# Media móvil
def moving_average(data, window=5):
    return data.rolling(window=window).mean()

# Gráfica de tubos
plt.figure(figsize=(12, 5))
plt.plot(df["Episodio"], df["Tubos Pasados"], label="Tubos Pasados", alpha=0.6)
plt.plot(df["Episodio"], moving_average(df["Tubos Pasados"]), label="Media móvil (5)", linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Tubos")
plt.title("🎯 Tubos pasados por episodio")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("grafica_tubos_pasados.png")
plt.show()

# Gráfica de recompensa
plt.figure(figsize=(12, 5))
plt.plot(df["Episodio"], df["Recompensa Total"], label="Recompensa Total", alpha=0.6)
plt.plot(df["Episodio"], moving_average(df["Recompensa Total"]), label="Media móvil (5)", linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("💰 Recompensa por episodio")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("grafica_recompensa_total.png")
plt.show()