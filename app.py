import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
import json

from flask import Flask, render_template, request, redirect, url_for, flash, Response

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medi_collect_secret_2026")

DB_NAME = "database.db"

# ==========================
# Paramètres (listes)
# ==========================
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

GROUPES_SANGUINS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
STATUTS = ["En cours", "Guéri", "Transféré", "Décédé"]


# ==========================
# DB helpers
# ==========================
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            age INTEGER NOT NULL,
            sexe TEXT NOT NULL,
            groupe_sanguin TEXT,
            maladie TEXT NOT NULL,
            service TEXT NOT NULL,
            temperature REAL NOT NULL,
            tension_systolique INTEGER,
            tension_diastolique INTEGER,
            poids REAL,
            duree_hospitalisation INTEGER NOT NULL,
            cout_traitement REAL NOT NULL,
            statut TEXT DEFAULT 'En cours',
            date_admission TEXT DEFAULT (date('now')),
            date_enregistrement TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Seed si vide (données réalistes) -> robustesse pour la démo en ligne
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if count == 0:
        sample_data = [
            ("Kamga Jean", 33, "Homme", "A+", "Grippe", "Médecine Générale", 37.6, 120, 80, 72.5, 14, 30000, "Guéri", "2026-03-01"),
            ("Nguemo Marie", 54, "Femme", "B+", "Diabète", "Endocrinologie", 37.0, 140, 90, 85.0, 60, 100000, "En cours", "2026-02-15"),
            ("Fouda Paul", 62, "Homme", "O+", "Insuffisance rénale", "Néphrologie", 36.5, 160, 100, 78.0, 86, 73000, "En cours", "2026-01-20"),
            ("Bella Rose", 1, "Femme", "AB+", "Jaunisse", "Pédiatrie", 38.6, 90, 60, 4.2, 5, 6000, "Guéri", "2026-04-10"),
            ("Mbarga Alain", 28, "Homme", "A-", "Grippe", "Médecine Générale", 39.2, 125, 82, 80.0, 7, 28000, "Guéri", "2026-04-01"),
            ("Tchoupo Eric", 45, "Homme", "O-", "Angine bactérienne", "ORL", 38.5, 130, 85, 90.5, 5, 21000, "Guéri", "2026-03-25"),
            ("Onana David", 12, "Homme", "B-", "Varicelle", "Pédiatrie", 38.1, 100, 65, 35.0, 10, 13000, "Guéri", "2026-03-28"),
            ("Ngo Bassa Aline", 35, "Femme", "A+", "Paludisme", "Médecine Générale", 39.8, 110, 70, 62.0, 4, 15000, "Guéri", "2026-04-05"),
            ("Essomba Pierre", 70, "Homme", "O+", "Hypertension", "Cardiologie", 37.2, 180, 110, 88.0, 30, 85000, "En cours", "2026-02-28"),
            ("Atangana Solange", 42, "Femme", "AB-", "Typhoïde", "Médecine Générale", 40.1, 115, 75, 58.0, 12, 25000, "Guéri", "2026-03-15"),
            ("Mvondo Jacques", 8, "Homme", "B+", "Bronchite", "Pédiatrie", 38.3, 95, 60, 25.0, 7, 18000, "Guéri", "2026-04-08"),
            ("Eyinga Claire", 55, "Femme", "A+", "Diabète", "Endocrinologie", 37.1, 145, 92, 95.0, 45, 120000, "En cours", "2026-01-10"),
            ("Biya Samuel", 38, "Homme", "O+", "Paludisme", "Médecine Générale", 39.5, 118, 78, 74.0, 5, 12000, "Guéri", "2026-04-12"),
            ("Fotso Brigitte", 25, "Femme", "B-", "Anémie", "Hématologie", 36.8, 105, 68, 52.0, 20, 45000, "En cours", "2026-03-20"),
            ("Tchinda Robert", 67, "Homme", "A-", "AVC", "Neurologie", 37.4, 175, 105, 82.0, 90, 250000, "En cours", "2026-01-05"),
            ("Mbouh Sandrine", 30, "Femme", "O+", "Grippe", "Médecine Générale", 38.8, 112, 72, 65.0, 3, 8000, "Guéri", "2026-04-15"),
            ("Njoya Ibrahim", 48, "Homme", "AB+", "Ulcère", "Gastroentérologie", 37.3, 128, 84, 76.0, 15, 35000, "Guéri", "2026-03-10"),
            ("Kemogne Diane", 22, "Femme", "A+", "Paludisme", "Médecine Générale", 39.9, 108, 70, 55.0, 6, 14000, "Guéri", "2026-04-18"),
            ("Tagne Michel", 58, "Homme", "B+", "Pneumonie", "Pneumologie", 39.0, 135, 88, 70.0, 21, 65000, "Guéri", "2026-02-20"),
            ("Ngono Patricia", 40, "Femme", "O-", "Hypertension", "Cardiologie", 37.0, 165, 98, 92.0, 25, 70000, "En cours", "2026-03-05"),
        ]
        conn.executemany("""
            INSERT INTO patients (
                nom, age, sexe, groupe_sanguin, maladie, service,
                temperature, tension_systolique, tension_diastolique, poids,
                duree_hospitalisation, cout_traitement, statut, date_admission
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        conn.commit()

    conn.close()


init_db()


# ==========================
# Utils
# ==========================
def validate_patient_form(form):
    errors = []

    nom = (form.get("nom") or "").strip()
    age = (form.get("age") or "").strip()
    sexe = (form.get("sexe") or "").strip()
    groupe_sanguin = (form.get("groupe_sanguin") or "").strip()
    maladie = (form.get("maladie") or "").strip()
    service = (form.get("service") or "").strip()
    temperature = (form.get("temperature") or "").strip()
    tension_sys = (form.get("tension_systolique") or "").strip()
    tension_dia = (form.get("tension_diastolique") or "").strip()
    poids = (form.get("poids") or "").strip()
    duree = (form.get("duree") or "").strip()
    cout = (form.get("cout") or "").strip()

    if not nom:
        errors.append("Le nom est obligatoire.")

    if not age.isdigit() or not (0 <= int(age) <= 120):
        errors.append("L'âge doit être un entier entre 0 et 120.")

    if sexe not in ["Homme", "Femme"]:
        errors.append("Veuillez choisir un sexe valide.")

    if groupe_sanguin and groupe_sanguin not in GROUPES_SANGUINS:
        errors.append("Groupe sanguin invalide.")

    if not maladie:
        errors.append("La maladie est obligatoire.")

    if not service:
        errors.append("Le service est obligatoire.")

    try:
        temp_val = float(temperature)
        if not (34 <= temp_val <= 43):
            errors.append("La température doit être entre 34°C et 43°C.")
    except Exception:
        errors.append("Température invalide.")

    # optionnels
    tension_sys_val = None
    tension_dia_val = None
    poids_val = None

    if tension_sys:
        try:
            tension_sys_val = int(tension_sys)
            if not (60 <= tension_sys_val <= 250):
                errors.append("Tension systolique hors intervalle (60-250).")
        except Exception:
            errors.append("Tension systolique invalide.")

    if tension_dia:
        try:
            tension_dia_val = int(tension_dia)
            if not (40 <= tension_dia_val <= 150):
                errors.append("Tension diastolique hors intervalle (40-150).")
        except Exception:
            errors.append("Tension diastolique invalide.")

    if poids:
        try:
            poids_val = float(poids)
            if not (0.5 <= poids_val <= 300):
                errors.append("Poids hors intervalle (0.5-300 kg).")
        except Exception:
            errors.append("Poids invalide.")

    try:
        duree_val = int(duree)
        if not (1 <= duree_val <= 365):
            errors.append("Durée hors intervalle (1-365 jours).")
    except Exception:
        errors.append("Durée invalide.")

    try:
        cout_val = float(cout)
        if cout_val < 0:
            errors.append("Le coût ne peut pas être négatif.")
    except Exception:
        errors.append("Coût invalide.")

    cleaned = dict(
        nom=nom,
        age=int(age) if age.isdigit() else None,
        sexe=sexe,
        groupe_sanguin=groupe_sanguin if groupe_sanguin else None,
        maladie=maladie,
        service=service,
        temperature=float(temperature) if temperature else None,
        tension_systolique=tension_sys_val,
        tension_diastolique=tension_dia_val,
        poids=poids_val,
        duree_hospitalisation=int(duree) if duree else None,
        cout_traitement=float(cout) if cout else None,
    )
    return errors, cleaned


# ==========================
# Routes
# ==========================
@app.route("/")
def index():
    return render_template(
        "index.html",
        maladies=MALADIES,
        services=SERVICES,
        groupes_sanguins=GROUPES_SANGUINS
    )


@app.route("/add", methods=["POST"])
def add_patient():
    errors, data = validate_patient_form(request.form)
    if errors:
        for e in errors:
            flash(e, "danger")
        return redirect(url_for("index"))

    conn = get_db()
    conn.execute("""
        INSERT INTO patients (
            nom, age, sexe, groupe_sanguin, maladie, service,
            temperature, tension_systolique, tension_diastolique, poids,
            duree_hospitalisation, cout_traitement
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["nom"], data["age"], data["sexe"], data["groupe_sanguin"], data["maladie"], data["service"],
        data["temperature"], data["tension_systolique"], data["tension_diastolique"], data["poids"],
        data["duree_hospitalisation"], data["cout_traitement"]
    ))
    conn.commit()
    conn.close()

    flash(f'✅ Patient "{data["nom"]}" enregistré avec succès.', "success")
    return redirect(url_for("data"))


@app.route("/data")
def data():
    conn = get_db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("data.html", patients=patients, total=len(patients))


@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):
    conn = get_db()

    if request.method == "POST":
        # validation légère (on réutilise le validateur puis on ajoute statut)
        errors, cleaned = validate_patient_form(request.form)
        statut = request.form.get("statut", "En cours")
        if statut not in STATUTS:
            errors.append("Statut invalide.")

        if errors:
            for e in errors:
                flash(e, "danger")
            conn.close()
            return redirect(url_for("edit", id=id))

        conn.execute("""
            UPDATE patients
            SET nom=?, age=?, sexe=?, groupe_sanguin=?, maladie=?, service=?,
                temperature=?, tension_systolique=?, tension_diastolique=?, poids=?,
                duree_hospitalisation=?, cout_traitement=?, statut=?
            WHERE id=?
        """, (
            cleaned["nom"], cleaned["age"], cleaned["sexe"], cleaned["groupe_sanguin"],
            cleaned["maladie"], cleaned["service"], cleaned["temperature"],
            cleaned["tension_systolique"], cleaned["tension_diastolique"], cleaned["poids"],
            cleaned["duree_hospitalisation"], cleaned["cout_traitement"], statut, id
        ))
        conn.commit()
        conn.close()
        flash("✅ Patient modifié avec succès.", "success")
        return redirect(url_for("data"))

    patient = conn.execute("SELECT * FROM patients WHERE id=?", (id,)).fetchone()
    conn.close()

    if not patient:
        flash("❌ Patient non trouvé.", "danger")
        return redirect(url_for("data"))

    return render_template(
        "edit.html",
        patient=patient,
        maladies=MALADIES,
        services=SERVICES,
        groupes_sanguins=GROUPES_SANGUINS,
        statuts=STATUTS
    )


@app.route("/delete/<int:id>")
def delete(id):
    conn = get_db()
    patient = conn.execute("SELECT nom FROM patients WHERE id=?", (id,)).fetchone()
    conn.execute("DELETE FROM patients WHERE id=?", (id,))
    conn.commit()
    conn.close()

    if patient:
        flash(f'🗑️ Patient "{patient["nom"]}" supprimé.', "warning")
    return redirect(url_for("data"))


@app.route("/export")
def export_csv():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)
    conn.close()

    csv_data = df.to_csv(index=False)
    filename = f"patients_{datetime.now():%Y%m%d_%H%M}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/analyse")
def analyse():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    if df.empty:
        return render_template("analyse.html", empty=True)

    total = len(df)
    hommes = int((df["sexe"] == "Homme").sum())
    femmes = int((df["sexe"] == "Femme").sum())
    en_cours = int((df["statut"] == "En cours").sum())
    gueris = int((df["statut"] == "Guéri").sum())

    # ---- Stats quantitatives (descriptives) ----
    quant = {
        "age": "Âge (ans)",
        "temperature": "Température (°C)",
        "duree_hospitalisation": "Durée hospitalisation (jours)",
        "cout_traitement": "Coût traitement (FCFA)",
        "poids": "Poids (kg)",
    }

    stats_table = []
    for col, label in quant.items():
        s = df[col].dropna()
        if len(s) == 0:
            continue
        mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else np.nan
        cv = (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan
        stats_table.append({
            "variable": label,
            "n": int(len(s)),
            "moyenne": round(float(s.mean()), 2),
            "mediane": round(float(s.median()), 2),
            "mode": round(float(mode_val), 2) if not pd.isna(mode_val) else "-",
            "ecart_type": round(float(s.std()), 2),
            "variance": round(float(s.var()), 2),
            "minimum": round(float(s.min()), 2),
            "maximum": round(float(s.max()), 2),
            "etendue": round(float(s.max() - s.min()), 2),
            "q1": round(float(s.quantile(0.25)), 2),
            "q3": round(float(s.quantile(0.75)), 2),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
            "cv": round(float(cv), 2) if not pd.isna(cv) else "-",
            "skewness": round(float(s.skew()), 3),
            "kurtosis": round(float(s.kurtosis()), 3),
        })

    # ---- Graphiques Plotly ----
    graphs = {}

    maladie_counts = df["maladie"].value_counts()
    fig_maladie = px.bar(
        x=maladie_counts.index, y=maladie_counts.values,
        labels={"x": "Maladie", "y": "Nombre de patients"},
        title="📊 Répartition des patients par maladie",
        color=maladie_counts.values,
        color_continuous_scale="Teal"
    )
    fig_maladie.update_layout(xaxis_tickangle=-40, height=450, plot_bgcolor="rgba(0,0,0,0)")
    graphs["maladie"] = json.dumps(fig_maladie, cls=plotly.utils.PlotlyJSONEncoder)

    fig_sexe = px.pie(
        df, names="sexe",
        title="👥 Répartition par sexe",
        hole=0.35,
        color_discrete_sequence=["#1a73e8", "#e91e63"]
    )
    fig_sexe.update_layout(height=380)
    graphs["sexe"] = json.dumps(fig_sexe, cls=plotly.utils.PlotlyJSONEncoder)

    fig_age = px.histogram(
        df, x="age", nbins=15,
        title="📈 Distribution des âges",
        marginal="box",
        color_discrete_sequence=["#667eea"]
    )
    fig_age.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)")
    graphs["age"] = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)

    fig_cout_duree = px.scatter(
        df, x="duree_hospitalisation", y="cout_traitement",
        color="service", size="age",
        hover_data=["nom", "maladie", "statut"],
        title="💰 Coût vs Durée d'hospitalisation (avec tendance)",
        trendline="ols",
        labels={"duree_hospitalisation": "Durée (jours)", "cout_traitement": "Coût (FCFA)"}
    )
    fig_cout_duree.update_layout(height=520, plot_bgcolor="rgba(0,0,0,0)")
    graphs["cout_duree"] = json.dumps(fig_cout_duree, cls=plotly.utils.PlotlyJSONEncoder)

    service_counts = df["service"].value_counts()
    fig_service = px.bar(
        x=service_counts.values, y=service_counts.index, orientation="h",
        title="🏥 Patients par service",
        labels={"x": "Nombre", "y": "Service"},
        color=service_counts.values,
        color_continuous_scale="Viridis"
    )
    fig_service.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    graphs["service"] = json.dumps(fig_service, cls=plotly.utils.PlotlyJSONEncoder)

    fig_statut = px.pie(
        df, names="statut", title="📋 Statut des patients",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_statut.update_layout(height=380)
    graphs["statut"] = json.dumps(fig_statut, cls=plotly.utils.PlotlyJSONEncoder)

    cout_service = df.groupby("service")["cout_traitement"].mean().sort_values()
    fig_cout_service = px.bar(
        x=cout_service.values, y=cout_service.index, orientation="h",
        title="💵 Coût moyen par service",
        labels={"x": "Coût moyen (FCFA)", "y": "Service"},
        color=cout_service.values,
        color_continuous_scale="Reds"
    )
    fig_cout_service.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    graphs["cout_service"] = json.dumps(fig_cout_service, cls=plotly.utils.PlotlyJSONEncoder)

    corr_cols = ["age", "temperature", "duree_hospitalisation", "cout_traitement"]
    corr_matrix = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix.values,
        x=["Âge", "Température", "Durée", "Coût"],
        y=["Âge", "Température", "Durée", "Coût"],
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="🔗 Matrice de corrélation",
        aspect="auto"
    )
    fig_corr.update_layout(height=420)
    graphs["correlation"] = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)

    # ---- Interprétations ----
    corr_dc = float(df["duree_hospitalisation"].corr(df["cout_traitement"]))
    if corr_dc > 0.7:
        interp_corr = f"Forte corrélation positive (r = {corr_dc:.2f}) : durée ↑ → coût ↑."
    elif corr_dc > 0.3:
        interp_corr = f"Corrélation modérée positive (r = {corr_dc:.2f})."
    else:
        interp_corr = f"Corrélation faible (r = {corr_dc:.2f})."

    interpretations = {
        "age_moyen": round(float(df["age"].mean()), 1),
        "temp_moyenne": round(float(df["temperature"].mean()), 1),
        "duree_moyenne": round(float(df["duree_hospitalisation"].mean()), 1),
        "cout_moyen": f'{df["cout_traitement"].mean():,.0f}',
        "cout_total": f'{df["cout_traitement"].sum():,.0f}',
        "top_maladie": f"{maladie_counts.index[0]} ({int(maladie_counts.values[0])} cas)",
        "top_service": service_counts.index[0],
        "taux_guerison": round((gueris / total) * 100, 1) if total else 0,
        "corr": interp_corr,
    }

    # ---- Fréquences qualitatives ----
    freq_tables = {}
    for col in ["sexe", "maladie", "service", "groupe_sanguin", "statut"]:
        vc = df[col].fillna("Non renseigné").value_counts()
        freq_tables[col] = [
            {"modalite": str(m), "effectif": int(n), "frequence": round(n / total * 100, 1)}
            for m, n in zip(vc.index, vc.values)
        ]

    return render_template(
        "analyse.html",
        empty=False,
        total=total,
        hommes=hommes,
        femmes=femmes,
        en_cours=en_cours,
        gueris=gueris,
        stats_table=stats_table,
        graphs=graphs,
        interpretations=interpretations,
        freq_tables=freq_tables
    )


# ==========================
# Render / Prod start
# ==========================
# En local: flask run ou python app.py
# En prod: gunicorn app:app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
