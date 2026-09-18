import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Domains
# ---------------------------------------------------------
D_MIN, D_MAX = 1000, 5000   # Demand
I_MIN, I_MAX = 100, 600     # Inventory
P_MIN, P_MAX = 2000, 7000   # Production

demand0 = 4000
inv0 = 300

# ---------------------------------------------------------
# Membership functions (linear, standard Tsukamoto-style FIS)
# ---------------------------------------------------------
def mu_demand_turun(x):
    if x <= D_MIN: return 1.0
    if x >= D_MAX: return 0.0
    return (D_MAX - x) / (D_MAX - D_MIN)

def mu_demand_naik(x):
    if x <= D_MIN: return 0.0
    if x >= D_MAX: return 1.0
    return (x - D_MIN) / (D_MAX - D_MIN)

def mu_inv_sedikit(y):
    if y <= I_MIN: return 1.0
    if y >= I_MAX: return 0.0
    return (I_MAX - y) / (I_MAX - I_MIN)

def mu_inv_banyak(y):
    if y <= I_MIN: return 0.0
    if y >= I_MAX: return 1.0
    return (y - I_MIN) / (I_MAX - I_MIN)

def mu_prod_turun(z):
    if z <= P_MIN: return 1.0
    if z >= P_MAX: return 0.0
    return (P_MAX - z) / (P_MAX - P_MIN)

def mu_prod_naik(z):
    if z <= P_MIN: return 0.0
    if z >= P_MAX: return 1.0
    return (z - P_MIN) / (P_MAX - P_MIN)

# inverse of production membership (solve z given alpha), since both are monotonic linear
def z_turun(alpha):
    return P_MAX - alpha * (P_MAX - P_MIN)

def z_naik(alpha):
    return P_MIN + alpha * (P_MAX - P_MIN)

# ---------------------------------------------------------
# Step 1: Fuzzification
# ---------------------------------------------------------
mu_D_turun = mu_demand_turun(demand0)
mu_D_naik = mu_demand_naik(demand0)
mu_I_sedikit = mu_inv_sedikit(inv0)
mu_I_banyak = mu_inv_banyak(inv0)

print("=== Fuzzification ===")
print(f"Demand={demand0}: mu_TURUN={mu_D_turun:.4f}, mu_NAIK={mu_D_naik:.4f}")
print(f"Inventory={inv0}: mu_SEDIKIT={mu_I_sedikit:.4f}, mu_BANYAK={mu_I_banyak:.4f}")

# ---------------------------------------------------------
# Step 2: Rule evaluation (AND -> min)
# ---------------------------------------------------------
alpha1 = min(mu_D_turun, mu_I_banyak)     # R1: Demand DECREASES & Inventory MANY -> Production DECREASES
alpha2 = min(mu_D_turun, mu_I_sedikit)    # R2: Demand DECREASES & Inventory FEW  -> Production DECREASES
alpha3 = min(mu_D_naik, mu_I_banyak)      # R3: Demand INCREASES & Inventory MANY -> Production INCREASES
alpha4 = min(mu_D_naik, mu_I_sedikit)     # R4: Demand INCREASES & Inventory FEW  -> Production INCREASES

print("\n=== Rule firing strengths (alpha) ===")
print(f"alpha1 (R1) = min({mu_D_turun:.4f}, {mu_I_banyak:.4f}) = {alpha1:.4f}")
print(f"alpha2 (R2) = min({mu_D_turun:.4f}, {mu_I_sedikit:.4f}) = {alpha2:.4f}")
print(f"alpha3 (R3) = min({mu_D_naik:.4f}, {mu_I_banyak:.4f}) = {alpha3:.4f}")
print(f"alpha4 (R4) = min({mu_D_naik:.4f}, {mu_I_sedikit:.4f}) = {alpha4:.4f}")

# ---------------------------------------------------------
# Step 3: Tsukamoto crisp output per rule (solve monotonic consequent)
# ---------------------------------------------------------
z1 = z_turun(alpha1)
z2 = z_turun(alpha2)
z3 = z_naik(alpha3)
z4 = z_naik(alpha4)

print("\n=== Per-rule crisp production (z_i) ===")
print(f"z1 (R1, TURUN) = 7000 - {alpha1:.4f}*5000 = {z1:.2f}")
print(f"z2 (R2, TURUN) = 7000 - {alpha2:.4f}*5000 = {z2:.2f}")
print(f"z3 (R3, NAIK)  = 2000 + {alpha3:.4f}*5000 = {z3:.2f}")
print(f"z4 (R4, NAIK)  = 2000 + {alpha4:.4f}*5000 = {z4:.2f}")

# ---------------------------------------------------------
# Step 4: Weighted average defuzzification
# ---------------------------------------------------------
num = alpha1*z1 + alpha2*z2 + alpha3*z3 + alpha4*z4
den = alpha1 + alpha2 + alpha3 + alpha4
Z = num/den

print("\n=== Defuzzification (weighted average) ===")
print(f"numerator sum(alpha_i*z_i) = {num:.2f}")
print(f"denominator sum(alpha_i)   = {den:.4f}")
print(f"Z = {Z:.2f}")

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
plt.rcParams.update({"font.size": 10})

# --- Demand plot ---
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.linspace(D_MIN, D_MAX, 400)
ax.plot(x, [mu_demand_turun(v) for v in x], label="DECREASES (Turun)", color="#1f77b4", lw=2)
ax.plot(x, [mu_demand_naik(v) for v in x], label="INCREASES (Naik)", color="#d62728", lw=2)
ax.axvline(demand0, color="gray", ls="--", lw=1)
ax.plot([demand0, demand0], [0, mu_D_turun], color="#1f77b4", ls=":", lw=1.5)
ax.plot([demand0, demand0], [0, mu_D_naik], color="#d62728", ls=":", lw=1.5)
ax.scatter([demand0, demand0], [mu_D_turun, mu_D_naik], color=["#1f77b4", "#d62728"], zorder=5)
ax.annotate(f"{mu_D_turun:.2f}", (demand0, mu_D_turun), textcoords="offset points", xytext=(8, 4))
ax.annotate(f"{mu_D_naik:.2f}", (demand0, mu_D_naik), textcoords="offset points", xytext=(8, -12))
ax.set_xlabel("Demand (packages/day)")
ax.set_ylabel("Membership degree")
ax.set_title("Fuzzy Sets for Demand")
ax.set_xlim(D_MIN, D_MAX)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="center left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("mf_demand.png", dpi=180)
plt.close(fig)

# --- Inventory plot ---
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.linspace(I_MIN, I_MAX, 400)
ax.plot(x, [mu_inv_sedikit(v) for v in x], label="FEW (Sedikit)", color="#1f77b4", lw=2)
ax.plot(x, [mu_inv_banyak(v) for v in x], label="MANY (Banyak)", color="#d62728", lw=2)
ax.axvline(inv0, color="gray", ls="--", lw=1)
ax.plot([inv0, inv0], [0, mu_I_sedikit], color="#1f77b4", ls=":", lw=1.5)
ax.plot([inv0, inv0], [0, mu_I_banyak], color="#d62728", ls=":", lw=1.5)
ax.scatter([inv0, inv0], [mu_I_sedikit, mu_I_banyak], color=["#1f77b4", "#d62728"], zorder=5)
ax.annotate(f"{mu_I_sedikit:.2f}", (inv0, mu_I_sedikit), textcoords="offset points", xytext=(8, 4))
ax.annotate(f"{mu_I_banyak:.2f}", (inv0, mu_I_banyak), textcoords="offset points", xytext=(8, -12))
ax.set_xlabel("Inventory (packages/day)")
ax.set_ylabel("Membership degree")
ax.set_title("Fuzzy Sets for Inventory")
ax.set_xlim(I_MIN, I_MAX)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="center left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("mf_inventory.png", dpi=180)
plt.close(fig)

# --- Production plot with per-rule alpha cuts ---
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.linspace(P_MIN, P_MAX, 400)
ax.plot(x, [mu_prod_turun(v) for v in x], label="DECREASES (Turun)", color="#1f77b4", lw=2)
ax.plot(x, [mu_prod_naik(v) for v in x], label="INCREASES (Naik)", color="#d62728", lw=2)

for z_i, a_i, label, color in [
    (z1, alpha1, "z1 (R1)", "#1f77b4"),
    (z2, alpha2, "z2 (R2)", "#1f77b4"),
    (z3, alpha3, "z3 (R3)", "#d62728"),
    (z4, alpha4, "z4 (R4)", "#d62728"),
]:
    ax.plot([z_i, z_i], [0, a_i], color=color, ls=":", lw=1.3)
    ax.scatter([z_i], [a_i], color=color, zorder=5, s=25)

ax.axvline(Z, color="green", lw=2, label=f"Z (final) = {Z:.1f}")
ax.set_xlabel("Production (packages/day)")
ax.set_ylabel("Membership degree")
ax.set_title("Fuzzy Sets for Production and Rule Outputs")
ax.set_xlim(P_MIN, P_MAX)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("mf_production.png", dpi=180)
plt.close(fig)

print("\nPlots saved: mf_demand.png, mf_inventory.png, mf_production.png")
print(f"\n>>> RECOMMENDED PRODUCTION Z = {Z:.2f} packages/day <<<")
