import os
import sqlite3
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, Response
)

app = Flask(__name__)
app.secret_key = "medicollect_2026"
DB = "database.db"

# ─────────────────────────────────────────────
# LISTES MÉTIER
# ─────────────────────────────────────────────
MALADIES = [
    "Grippe", "Paludisme", "Typhoïde", "Diabète", "Hypertension",
    "Pneumonie", "Bronchite", "Varicelle", "Jaunisse", "Anémie",
    "Insuffisance rénale", "AVC", "Ulcère", "Angine bactérienne",
    "Tuberculose", "Fièvre", "Autre"
]
SERVICES = [
    "Médecine Générale", "Pédiatrie", "Cardiologie", "Neurologie",
    "Endocrinologie", "Néphrologie", "ORL", "Pneumologie",
    "Gastroentérologie", "Hématologie", "Chirurgie", "Urgences"
]
GROUPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
STATUTS = ["En cours", "Guéri", "Transféré", "Décédé"]

# ─────────────────────────────────────────────
# BASE DE DONNÉES
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            nom                  TEXT    NOT NULL,
            age                  INTEGER NOT NULL,
            sexe                 TEXT    NOT NULL,
            groupe_sanguin       TEXT,
            maladie              TEXT    NOT NULL,
            service              TEXT    NOT NULL,
            temperature          REAL    NOT NULL,
            tension_systolique   INTEGER,
            tension_diastolique  INTEGER,
            poids                REAL,
            duree                INTEGER NOT NULL,
            cout                 REAL    NOT NULL,
            statut               TEXT    NOT NULL DEFAULT 'En cours',
            date_admission       TEXT    NOT NULL DEFAULT (date('now'))
        )
    """)
    conn.commit()

    if conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0:
        seed = [
            ("Kamga Jean",      33, "Homme", "A+",  "Grippe",            "Médecine Générale", 37.6, 120, 80,  72.5, 14, 30000,  "Guéri",    "2026-03-01"),
            ("Nguemo Marie",    54, "Femme", "B+",  "Diabète",           "Endocrinologie",    37.0, 140, 90,  85.0, 60, 100000, "En cours", "2026-02-15"),
            ("Fouda Paul",      62, "Homme", "O+",  "Insuffisance rénale","Néphrologie",       36.5, 160, 100, 78.0, 86, 73000,  "En cours", "2026-01-20"),
            ("Bella Rose",       1, "Femme", "AB+", "Jaunisse",          "Pédiatrie",         38.6, 90,  60,  4.2,  5,  6000,   "Guéri",    "2026-04-10"),
            ("Mbarga Alain",    28, "Homme", "A-",  "Grippe",            "Médecine Générale", 39.2, 125, 82,  80.0, 7,  28000,  "Guéri",    "2026-04-01"),
            ("Tchoupo Eric",    45, "Homme", "O-",  "Angine bactérienne","ORL",               38.5, 130, 85,  90.5, 5,  21000,  "Guéri",    "2026-03-25"),
            ("Onana David",     12, "Homme", "B-",  "Varicelle",         "Pédiatrie",         38.1, 100, 65,  35.0, 10, 13000,  "Guéri",    "2026-03-28"),
            ("Ngo Bassa Aline", 35, "Femme", "A+",  "Paludisme",         "Médecine Générale", 39.8, 110, 70,  62.0, 4,  15000,  "Guéri",    "2026-04-05"),
            ("Essomba Pierre",  70, "Homme", "O+",  "Hypertension",      "Cardiologie",       37.2, 180, 110, 88.0, 30, 85000,  "En cours", "2026-02-28"),
            ("Atangana Solange",42, "Femme", "AB-", "Typhoïde",          "Médecine Générale", 40.1, 115, 75,  58.0, 12, 25000,  "Guéri",    "2026-03-15"),
            ("Mvondo Jacques",   8, "Homme", "B+",  "Bronchite",         "Pédiatrie",         38.3, 95,  60,  25.0, 7,  18000,  "Guéri",    "2026-04-08"),
            ("Eyinga Claire",   55, "Femme", "A+",  "Diabète",           "Endocrinologie",    37.1, 145, 92,  95.0, 45, 120000, "En cours", "2026-01-10"),
            ("Biya Samuel",     38, "Homme", "O+",  "Paludisme",         "Médecine Générale", 39.5, 118, 78,  74.0, 5,  12000,  "Guéri",    "2026-04-12"),
            ("Fotso Brigitte",  25, "Femme", "B-",  "Anémie",            "Hématologie",       36.8, 105, 68,  52.0, 20, 45000,  "En cours", "2026-03-20"),
            ("Tchinda Robert",  67, "Homme", "A-",  "AVC",               "Neurologie",        37.4, 175, 105, 82.0, 90, 250000, "En cours", "2026-01-05"),
            ("Mbouh Sandrine",  30, "Femme", "O+",  "Grippe",            "Médecine Générale", 38.8, 112, 72,  65.0, 3,  8000,   "Guéri",    "2026-04-15"),
            ("Njoya Ibrahim",   48, "Homme", "AB+", "Ulcère",            "Gastroentérologie", 37.3, 128, 84,  76.0, 15, 35000,  "Guéri",    "2026-03-10"),
            ("Kemogne Diane",   22, "Femme", "A+",  "Paludisme",         "Médecine Générale", 39.9, 108, 70,  55.0, 6,  14000,  "Guéri",    "2026-04-18"),
            ("Tagne Michel",    58, "Homme", "B+",  "Pneumonie",         "Pneumologie",       39.0, 135, 88,  70.0, 21, 65000,  "Guéri",    "2026-02-20"),
            ("Ngono Patricia",  40, "Femme", "O-",  "Hypertension",      "Cardiologie",       37.0, 165, 98,  92.0, 25, 70000,  "En cours", "2026-03-05"),
        ]
        conn.executemany("""
            INSERT INTO patients
                (nom,age,sexe,groupe_sanguin,maladie,service,
                 temperature,tension_systolique,tension_diastolique,poids,
                 duree,cout,statut,date_admission)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, seed)
        conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def valider(form):
    err = []
    nom  = (form.get("nom") or "").strip()
    age  = (form.get("age") or "").strip()
    sexe = (form.get("sexe") or "").strip()
    gs   = (form.get("groupe_sanguin") or "").strip()
    mal  = (form.get("maladie") or "").strip()
    srv  = (form.get("service") or "").strip()
    temp = (form.get("temperature") or "").strip()
    tsys = (form.get("tension_systolique") or "").strip()
    tdia = (form.get("tension_diastolique") or "").strip()
    pds  = (form.get("poids") or "").strip()
    dur  = (form.get("duree") or "").strip()
    cout = (form.get("cout") or "").strip()

    if not nom:            err.append("Le nom est obligatoire.")
    if not age.isdigit() or not (0 <= int(age) <= 120):
        err.append("Âge invalide (0-120).")
    if sexe not in ["Homme", "Femme"]:
        err.append("Sexe invalide.")
    if not mal: err.append("Maladie obligatoire.")
    if not srv: err.append("Service obligatoire.")

    temp_v = None
    try:
        temp_v = float(temp)
        if not (34 <= temp_v <= 43): err.append("Température hors limite (34-43).")
    except: err.append("Température invalide.")

    tsys_v = tdia_v = pds_v = None
    if tsys:
        try: tsys_v = int(tsys)
        except: err.append("Tension systolique invalide.")
    if tdia:
        try: tdia_v = int(tdia)
        except: err.append("Tension diastolique invalide.")
    if pds:
        try: pds_v = float(pds)
        except: err.append("Poids invalide.")

    dur_v = None
    try:
        dur_v = int(dur)
        if dur_v < 1: err.append("Durée min 1 jour.")
    except: err.append("Durée invalide.")

    cout_v = None
    try:
        cout_v = float(cout)
        if cout_v < 0: err.append("Coût négatif interdit.")
    except: err.append("Coût invalide.")

    cleaned = dict(
        nom=nom, age=int(age) if age.isdigit() else 0,
        sexe=sexe, groupe_sanguin=gs or None,
        maladie=mal, service=srv,
        temperature=temp_v,
        tension_systolique=tsys_v, tension_diastolique=tdia_v,
        poids=pds_v, duree=dur_v, cout=cout_v
    )
    return err, cleaned

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
        maladies=MALADIES, services=SERVICES, groupes=GROUPES)

@app.route("/add", methods=["POST"])
def add():
    err, d = valider(request.form)
    if err:
        for e in err: flash(e, "danger")
        return redirect(url_for("index"))
    conn = get_db()
    conn.execute("""
        INSERT INTO patients
            (nom,age,sexe,groupe_sanguin,maladie,service,
             temperature,tension_systolique,tension_diastolique,poids,
             duree,cout,statut)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'En cours')
    """, (d["nom"],d["age"],d["sexe"],d["groupe_sanguin"],
          d["maladie"],d["service"],d["temperature"],
          d["tension_systolique"],d["tension_diastolique"],
          d["poids"],d["duree"],d["cout"]))
    conn.commit(); conn.close()
    flash(f"✅ Patient « {d['nom']} » enregistré.", "success")
    return redirect(url_for("data"))

@app.route("/data")
def data():
    conn = get_db()
    rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("data.html", patients=rows, total=len(rows))

@app.route("/edit/<int:pid>", methods=["GET","POST"])
def edit(pid):
    conn = get_db()
    if request.method == "POST":
        err, d = valider(request.form)
        statut = request.form.get("statut","En cours")
        if statut not in STATUTS: err.append("Statut invalide.")
        if err:
            for e in err: flash(e,"danger")
            conn.close()
            return redirect(url_for("edit", pid=pid))
        conn.execute("""
            UPDATE patients SET
                nom=?,age=?,sexe=?,groupe_sanguin=?,maladie=?,service=?,
                temperature=?,tension_systolique=?,tension_diastolique=?,
                poids=?,duree=?,cout=?,statut=?
            WHERE id=?
        """, (d["nom"],d["age"],d["sexe"],d["groupe_sanguin"],
              d["maladie"],d["service"],d["temperature"],
              d["tension_systolique"],d["tension_diastolique"],
              d["poids"],d["duree"],d["cout"],statut,pid))
        conn.commit(); conn.close()
        flash("✅ Patient modifié.", "success")
        return redirect(url_for("data"))

    p = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not p:
        flash("Patient introuvable.","danger")
        return redirect(url_for("data"))
    return render_template("edit.html", p=p,
        maladies=MALADIES, services=SERVICES,
        groupes=GROUPES, statuts=STATUTS)

@app.route("/delete/<int:pid>")
def delete(pid):
    conn = get_db()
    row = conn.execute("SELECT nom FROM patients WHERE id=?", (pid,)).fetchone()
    conn.execute("DELETE FROM patients WHERE id=?", (pid,))
    conn.commit(); conn.close()
    if row: flash(f"🗑️ Patient « {row['nom']} » supprimé.", "warning")
    return redirect(url_for("data"))

@app.route("/export")
def export():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id", conn)
    conn.close()
    return Response(df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition":
            f"attachment;filename=patients_{datetime.now():%Y%m%d_%H%M}.csv"})

# ─────────────────────────────────────────────
# ANALYSE
# ─────────────────────────────────────────────
@app.route("/analyse")
def analyse():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    if len(df) < 3:
        return render_template("analyse.html", empty=True,
            message="Il faut au moins 3 patients pour faire une analyse.")

    total   = len(df)
    hommes  = int((df["sexe"]=="Homme").sum())
    femmes  = int((df["sexe"]=="Femme").sum())
    gueris  = int((df["statut"]=="Guéri").sum())
    en_cours= int((df["statut"]=="En cours").sum())

    # ── Statistiques descriptives ──────────────────────
    cols_q = {
        "age":          "Âge (ans)",
        "temperature":  "Température (°C)",
        "duree":        "Durée hosp. (jours)",
        "cout":         "Coût (FCFA)",
        "poids":        "Poids (kg)",
    }
    stats = []
    for col, label in cols_q.items():
        s = df[col].dropna()
        if len(s) < 2: continue
        cv = s.std()/s.mean()*100 if s.mean() else 0
        mode_v = float(s.mode().iloc[0]) if len(s.mode()) else None
        stats.append(dict(
            variable=label, n=int(len(s)),
            moyenne=round(float(s.mean()),2),
            mediane=round(float(s.median()),2),
            mode=round(mode_v,2) if mode_v is not None else "-",
            ecart_type=round(float(s.std()),2),
            variance=round(float(s.var()),2),
            minimum=round(float(s.min()),2),
            maximum=round(float(s.max()),2),
            etendue=round(float(s.max()-s.min()),2),
            q1=round(float(s.quantile(.25)),2),
            q3=round(float(s.quantile(.75)),2),
            iqr=round(float(s.quantile(.75)-s.quantile(.25)),2),
            cv=round(cv,2),
            skewness=round(float(s.skew()),3),
            kurtosis=round(float(s.kurtosis()),3),
        ))

    # ── Tableaux de fréquences ─────────────────────────
    freq = {}
    for col in ["sexe","maladie","service","statut","groupe_sanguin"]:
        vc = df[col].fillna("Non renseigné").value_counts()
        freq[col] = [
            {"modalite": str(m), "effectif": int(n),
             "pct": round(n/total*100, 1)}
            for m,n in zip(vc.index, vc.values)
        ]

    # ── Régression linéaire simple (durée → coût) ──────
    dfr = df[["duree","cout","age","temperature"]].dropna()
    reg1 = {}
    if len(dfr) >= 3:
        X1 = dfr[["duree"]].values
        y1 = dfr["cout"].values
        m  = LinearRegression().fit(X1, y1)
        y_pred = m.predict(X1)
        r2  = r2_score(y1, y_pred)
        # statsmodels pour p-value
        Xsm = sm.add_constant(X1)
        res = sm.OLS(y1, Xsm).fit()
        reg1 = dict(
            coef=round(float(m.coef_[0]),2),
            intercept=round(float(m.intercept_),2),
            r2=round(r2,4),
            r2_pct=round(r2*100,1),
            pvalue=round(float(res.pvalues[1]),4),
            equation=f"Coût = {m.coef_[0]:.1f} × Durée + {m.intercept_:.0f}",
            interpretation=(
                f"La durée explique {r2*100:.1f}% de la variation du coût. "
                f"Chaque jour supplémentaire ajoute environ {m.coef_[0]:.0f} FCFA."
            )
        )
        # prédiction interactive (durée = 30 jours par défaut)
        pred_duree = 30
        reg1["pred_duree"] = pred_duree
        reg1["pred_cout"]  = round(float(m.predict([[pred_duree]])[0]), 0)

    # ── Régression linéaire multiple ───────────────────
    reg2 = {}
    if len(dfr) >= 5:
        X2 = dfr[["duree","age","temperature"]].values
        y2 = dfr["cout"].values
        m2 = LinearRegression().fit(X2, y2)
        r2m = r2_score(y2, m2.predict(X2))
        Xsm2 = sm.add_constant(X2)
        res2 = sm.OLS(y2, Xsm2).fit()
        reg2 = dict(
            coef_duree=round(float(m2.coef_[0]),2),
            coef_age=round(float(m2.coef_[1]),2),
            coef_temp=round(float(m2.coef_[2]),2),
            intercept=round(float(m2.intercept_),2),
            r2=round(r2m,4),
            r2_pct=round(r2m*100,1),
            equation=(
                f"Coût = {m2.coef_[0]:.1f}×Durée "
                f"+ {m2.coef_[1]:.1f}×Âge "
                f"+ {m2.coef_[2]:.1f}×Température "
                f"+ {m2.intercept_:.0f}"
            ),
            interpretation=(
                f"Le modèle multiple explique {r2m*100:.1f}% de la variation du coût. "
                f"La durée reste le facteur dominant ({m2.coef_[0]:.1f} FCFA/jour)."
            )
        )

    # ── Corrélations ───────────────────────────────────
    corr_dc = round(float(dfr["duree"].corr(dfr["cout"])),3)

    # ── Graphiques Plotly ─────────────────────────────
    graphs = {}

    def jdump(fig): return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 1 – Maladies
    vc_mal = df["maladie"].value_counts()
    graphs["maladie"] = jdump(px.bar(
        x=vc_mal.index, y=vc_mal.values,
        title="Répartition par maladie",
        labels={"x":"Maladie","y":"Nb"},
        color=vc_mal.values, color_continuous_scale="Teal"
    ).update_layout(height=420, xaxis_tickangle=-35,
                    plot_bgcolor="rgba(0,0,0,0)"))

    # 2 – Sexe
    graphs["sexe"] = jdump(px.pie(
        df, names="sexe", title="Répartition par sexe",
        hole=0.4, color_discrete_sequence=["#1a73e8","#e91e63"]
    ).update_layout(height=360))

    # 3 – Distribution des âges
    graphs["age"] = jdump(px.histogram(
        df, x="age", nbins=15, marginal="box",
        title="Distribution des âges",
        color_discrete_sequence=["#667eea"]
    ).update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)"))

    # 4 – Scatter durée vs coût avec droite de régression
    graphs["reg_simple"] = jdump(px.scatter(
        dfr, x="duree", y="cout", trendline="ols",
        title="Régression simple : Durée → Coût",
        labels={"duree":"Durée (jours)","cout":"Coût (FCFA)"},
        color_discrete_sequence=["#667eea"]
    ).update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)"))

    # 5 – Coût par service
    cs = df.groupby("service")["cout"].mean().sort_values()
    graphs["cout_service"] = jdump(px.bar(
        x=cs.values, y=cs.index, orientation="h",
        title="Coût moyen par service (FCFA)",
        labels={"x":"Coût moyen","y":"Service"},
        color=cs.values, color_continuous_scale="Reds"
    ).update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)"))

    # 6 – Statut
    graphs["statut"] = jdump(px.pie(
        df, names="statut", title="Statut des patients",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2
    ).update_layout(height=360))

    # 7 – Boxplot température par maladie
    graphs["temp"] = jdump(px.box(
        df, x="maladie", y="temperature",
        title="Température par maladie",
        color="maladie"
    ).update_layout(height=450, xaxis_tickangle=-35,
                    plot_bgcolor="rgba(0,0,0,0)", showlegend=False))

    # 8 – Corrélation
    corr_cols = ["age","temperature","duree","cout"]
    corr_mat  = df[corr_cols].corr()
    graphs["correlation"] = jdump(px.imshow(
        corr_mat.values,
        x=["Âge","Temp.","Durée","Coût"],
        y=["Âge","Temp.","Durée","Coût"],
        text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Matrice de corrélation"
    ).update_layout(height=400))

    # ── Résumé global ──────────────────────────────────
    resume = dict(
        total=total, hommes=hommes, femmes=femmes,
        gueris=gueris, en_cours=en_cours,
        taux_guerison=round(gueris/total*100,1),
        age_moy=round(float(df["age"].mean()),1),
        temp_moy=round(float(df["temperature"].mean()),1),
        duree_moy=round(float(df["duree"].mean()),1),
        cout_moy=int(df["cout"].mean()),
        cout_total=int(df["cout"].sum()),
        top_maladie=df["maladie"].value_counts().index[0],
        top_service=df["service"].value_counts().index[0],
        corr_dc=corr_dc,
    )

    return render_template("analyse.html",
        empty=False, resume=resume, stats=stats, freq=freq,
        reg1=reg1, reg2=reg2, graphs=graphs)

# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
