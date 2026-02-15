"""
Application Streamlit - Analyse Météo et Prix de l'Électricité en Allemagne
Interface interactive pour visualiser les corrélations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Analyse Météo & Électricité - Allemagne",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #e3f2fd;
    }
    </style>
""", unsafe_allow_html=True)

class MeteoElectriciteApp:
    """
    Application d'analyse météo et prix de l'électricité
    """
    
    def __init__(self):
        self.villes_allemandes = {
            'Berlin': {'lat': 52.52, 'lon': 13.405, 'region': 'Est'},
            'Munich': {'lat': 48.1351, 'lon': 11.582, 'region': 'Sud'},
            'Hambourg': {'lat': 53.5511, 'lon': 9.9937, 'region': 'Nord'},
            'Cologne': {'lat': 50.9375, 'lon': 6.9603, 'region': 'Ouest'},
            'Francfort': {'lat': 50.1109, 'lon': 8.6821, 'region': 'Centre'},
            'Stuttgart': {'lat': 48.7758, 'lon': 9.1829, 'region': 'Sud-Ouest'},
            'Düsseldorf': {'lat': 51.2277, 'lon': 6.7735, 'region': 'Ouest'},
            'Dortmund': {'lat': 51.5136, 'lon': 7.4653, 'region': 'Ouest'},
            'Leipzig': {'lat': 51.3397, 'lon': 12.3731, 'region': 'Est'},
            'Dresde': {'lat': 51.0504, 'lon': 13.7373, 'region': 'Est'}
        }
        
    def simuler_meteo_realiste(self, ville, lat, seed_offset=0):
        """
        Simule des données météo réalistes
        """
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)
        
        facteur_nord = (lat - 48) / 5
        
        # Température
        temp_base = 6.0
        temp = temp_base - facteur_nord * 3 + random.uniform(-2, 2)
        
        # Vent
        vent_base = 12.0
        vent = vent_base + facteur_nord * 8 + random.uniform(-3, 5)
        vent = max(vent, 0)
        
        # Nuages
        nuages = random.uniform(40, 90)
        
        return {
            'temperature_actuelle': round(temp, 1),
            'vitesse_vent': round(vent, 1),
            'couverture_nuageuse': round(nuages, 1)
        }
    
    def calculer_prix_electricite(self, meteo_data):
        """
        Calcule le prix de l'électricité
        """
        prix_base = 50.0
        
        # Effet du vent
        vitesse_vent = meteo_data['vitesse_vent']
        if vitesse_vent > 10:
            reduction_vent = min((vitesse_vent - 10) * 0.8, 20)
        else:
            reduction_vent = 0
        
        # Effet du soleil
        couverture = meteo_data['couverture_nuageuse']
        ensoleillement = 100 - couverture
        if ensoleillement > 30:
            reduction_solaire = (ensoleillement - 30) * 0.15
        else:
            reduction_solaire = 0
        
        # Effet de la température
        temp = meteo_data['temperature_actuelle']
        if temp < 5:
            augmentation_temp = (5 - temp) * 1.2
        elif temp < 10:
            augmentation_temp = (10 - temp) * 0.5
        elif temp > 25:
            augmentation_temp = (temp - 25) * 0.8
        else:
            augmentation_temp = 0
        
        prix_final = prix_base - reduction_vent - reduction_solaire + augmentation_temp
        prix_final = max(prix_final, 15)
        
        return {
            'prix_EUR_MWh': round(prix_final, 2),
            'reduction_vent': round(reduction_vent, 2),
            'reduction_solaire': round(reduction_solaire, 2),
            'augmentation_temp': round(augmentation_temp, 2)
        }
    
    def generer_donnees_historiques(self, ville, coords, jours=30):
        """
        Génère des données historiques sur plusieurs jours
        """
        donnees = []
        date_debut = datetime.now() - timedelta(days=jours)
        
        for i in range(jours):
            date = date_debut + timedelta(days=i)
            meteo = self.simuler_meteo_realiste(ville, coords['lat'], seed_offset=i)
            prix = self.calculer_prix_electricite(meteo)
            
            donnees.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Température': meteo['temperature_actuelle'],
                'Vent': meteo['vitesse_vent'],
                'Nuages': meteo['couverture_nuageuse'],
                'Prix': prix['prix_EUR_MWh']
            })
        
        return pd.DataFrame(donnees)
    
    def collecter_donnees_actuelles(self):
        """
        Collecte les données actuelles pour toutes les villes
        """
        resultats = []
        
        for ville, coords in self.villes_allemandes.items():
            meteo = self.simuler_meteo_realiste(ville, coords['lat'])
            prix = self.calculer_prix_electricite(meteo)
            
            resultats.append({
                'Ville': ville,
                'Région': coords['region'],
                'Température (°C)': meteo['temperature_actuelle'],
                'Vent (km/h)': meteo['vitesse_vent'],
                'Nuages (%)': meteo['couverture_nuageuse'],
                'Prix (EUR/MWh)': prix['prix_EUR_MWh'],
                'Économie Vent': -prix['reduction_vent'],
                'Économie Solaire': -prix['reduction_solaire'],
                'Surcoût Temp': prix['augmentation_temp']
            })
        
        return pd.DataFrame(resultats)

def main():
    """
    Fonction principale de l'application
    """
    app = MeteoElectriciteApp()
    
    # Header
    st.markdown('<div class="main-header">⚡ Analyse Météo & Prix de l\'Électricité - Allemagne 🇩🇪</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Mode de visualisation
    mode = st.sidebar.radio(
        "Mode d'analyse",
        ["📊 Vue d'ensemble", "🏙️ Analyse par ville", "📈 Tendances historiques", "🔍 Analyse approfondie"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **À propos**
    
    Cette application analyse la corrélation entre:
    - 🌡️ Température
    - 💨 Vitesse du vent
    - ☁️ Couverture nuageuse
    - ⚡ Prix de l'électricité
    
    **Données**: Simulées de manière réaliste
    """)
    
    # Collecter les données
    df_actuel = app.collecter_donnees_actuelles()
    
    # MODE 1: Vue d'ensemble
    if mode == "📊 Vue d'ensemble":
        st.header("📊 Vue d'ensemble - Situation actuelle")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prix moyen",
                f"{df_actuel['Prix (EUR/MWh)'].mean():.2f} €/MWh",
                delta=f"{df_actuel['Prix (EUR/MWh)'].std():.2f} écart-type"
            )
        
        with col2:
            ville_min = df_actuel.loc[df_actuel['Prix (EUR/MWh)'].idxmin(), 'Ville']
            prix_min = df_actuel['Prix (EUR/MWh)'].min()
            st.metric("Prix minimum", f"{prix_min:.2f} €/MWh", f"à {ville_min}")
        
        with col3:
            ville_max = df_actuel.loc[df_actuel['Prix (EUR/MWh)'].idxmax(), 'Ville']
            prix_max = df_actuel['Prix (EUR/MWh)'].max()
            st.metric("Prix maximum", f"{prix_max:.2f} €/MWh", f"à {ville_max}")
        
        with col4:
            st.metric(
                "Température moy.",
                f"{df_actuel['Température (°C)'].mean():.1f}°C",
                f"Vent: {df_actuel['Vent (km/h)'].mean():.1f} km/h"
            )
        
        st.markdown("---")
        
        # Graphiques principaux
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Prix par ville
            fig_prix = px.bar(
                df_actuel.sort_values('Prix (EUR/MWh)'),
                x='Prix (EUR/MWh)',
                y='Ville',
                orientation='h',
                title='Prix de l\'électricité par ville',
                color='Prix (EUR/MWh)',
                color_continuous_scale='RdYlGn_r',
                text='Prix (EUR/MWh)'
            )
            fig_prix.update_traces(texttemplate='%{text:.1f}€', textposition='outside')
            fig_prix.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_prix, use_container_width=True)
        
        with col_right:
            # Répartition régionale
            fig_region = px.pie(
                df_actuel,
                names='Région',
                values='Prix (EUR/MWh)',
                title='Répartition des prix par région',
                hole=0.4
            )
            fig_region.update_layout(height=500)
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Corrélations
        st.subheader("🔗 Analyse des corrélations")
        
        col_corr1, col_corr2 = st.columns(2)
        
        with col_corr1:
            # Température vs Prix
            fig_temp = px.scatter(
                df_actuel,
                x='Température (°C)',
                y='Prix (EUR/MWh)',
                size='Vent (km/h)',
                color='Nuages (%)',
                hover_data=['Ville'],
                title='Température vs Prix (taille = vent, couleur = nuages)',
                trendline='ols'
            )
            fig_temp.update_layout(height=400)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col_corr2:
            # Vent vs Prix
            fig_vent = px.scatter(
                df_actuel,
                x='Vent (km/h)',
                y='Prix (EUR/MWh)',
                size='Température (°C)',
                color='Ville',
                hover_data=['Ville'],
                title='Vent vs Prix (taille = température)',
                trendline='ols'
            )
            fig_vent.update_layout(height=400)
            st.plotly_chart(fig_vent, use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("📊 Matrice de corrélation")
        colonnes_num = ['Température (°C)', 'Vent (km/h)', 'Nuages (%)', 'Prix (EUR/MWh)']
        corr_matrix = df_actuel[colonnes_num].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.3f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Matrice de corrélation'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tableau des données
        st.subheader("📋 Données détaillées")
        st.dataframe(
            df_actuel.style.background_gradient(subset=['Prix (EUR/MWh)'], cmap='RdYlGn_r'),
            use_container_width=True
        )
    
    # MODE 2: Analyse par ville
    elif mode == "🏙️ Analyse par ville":
        st.header("🏙️ Analyse détaillée par ville")
        
        # Sélection de la ville
        ville_selectionnee = st.selectbox(
            "Choisissez une ville",
            options=list(app.villes_allemandes.keys()),
            index=0
        )
        
        coords = app.villes_allemandes[ville_selectionnee]
        donnees_ville = df_actuel[df_actuel['Ville'] == ville_selectionnee].iloc[0]
        
        # Métriques de la ville
        st.subheader(f"📍 {ville_selectionnee} - {coords['region']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Température", f"{donnees_ville['Température (°C)']}°C")
        
        with col2:
            st.metric("💨 Vent", f"{donnees_ville['Vent (km/h)']} km/h")
        
        with col3:
            st.metric("☁️ Nuages", f"{donnees_ville['Nuages (%)']}%")
        
        with col4:
            st.metric("⚡ Prix", f"{donnees_ville['Prix (EUR/MWh)']}€/MWh")
        
        st.markdown("---")
        
        # Décomposition du prix
        st.subheader("💰 Décomposition du prix")
        
        prix_base = 50.0
        impacts = pd.DataFrame({
            'Composant': [
                'Prix de base',
                'Économie éolienne',
                'Économie solaire',
                'Surcoût température',
                'Prix final'
            ],
            'Valeur': [
                prix_base,
                donnees_ville['Économie Vent'],
                donnees_ville['Économie Solaire'],
                donnees_ville['Surcoût Temp'],
                donnees_ville['Prix (EUR/MWh)']
            ],
            'Type': ['Base', 'Réduction', 'Réduction', 'Augmentation', 'Final']
        })
        
        fig_waterfall = go.Figure(go.Waterfall(
            x=impacts['Composant'],
            y=impacts['Valeur'],
            measure=['absolute', 'relative', 'relative', 'relative', 'total'],
            text=impacts['Valeur'].round(2),
            textposition='outside',
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
            decreasing={'marker': {'color': 'green'}},
            increasing={'marker': {'color': 'red'}},
            totals={'marker': {'color': 'blue'}}
        ))
        
        fig_waterfall.update_layout(
            title=f'Formation du prix à {ville_selectionnee}',
            yaxis_title='EUR/MWh',
            height=500
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Comparaison avec les autres villes
        st.subheader("📊 Comparaison avec les autres villes")
        
        # Rang de la ville
        rang_prix = (df_actuel['Prix (EUR/MWh)'] > donnees_ville['Prix (EUR/MWh)']).sum() + 1
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.info(f"""
            **Position**: {rang_prix}ème ville la moins chère sur 10
            
            **Écart avec la moyenne**: {donnees_ville['Prix (EUR/MWh)'] - df_actuel['Prix (EUR/MWh)'].mean():+.2f} €/MWh
            """)
        
        with col_comp2:
            # Graphique radar
            categories = ['Température', 'Vent', 'Nuages', 'Prix']
            
            # Normaliser les valeurs pour le radar (0-100)
            ville_vals = [
                (donnees_ville['Température (°C)'] - df_actuel['Température (°C)'].min()) / 
                (df_actuel['Température (°C)'].max() - df_actuel['Température (°C)'].min()) * 100,
                (donnees_ville['Vent (km/h)'] - df_actuel['Vent (km/h)'].min()) / 
                (df_actuel['Vent (km/h)'].max() - df_actuel['Vent (km/h)'].min()) * 100,
                (donnees_ville['Nuages (%)'] - df_actuel['Nuages (%)'].min()) / 
                (df_actuel['Nuages (%)'].max() - df_actuel['Nuages (%)'].min()) * 100,
                (donnees_ville['Prix (EUR/MWh)'] - df_actuel['Prix (EUR/MWh)'].min()) / 
                (df_actuel['Prix (EUR/MWh)'].max() - df_actuel['Prix (EUR/MWh)'].min()) * 100
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=ville_vals,
                theta=categories,
                fill='toself',
                name=ville_selectionnee
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=f'Profil de {ville_selectionnee} (normalisé)',
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # MODE 3: Tendances historiques
    elif mode == "📈 Tendances historiques":
        st.header("📈 Tendances historiques")
        
        # Sélection de la ville
        ville_hist = st.selectbox(
            "Choisissez une ville pour l'analyse historique",
            options=list(app.villes_allemandes.keys()),
            index=0
        )
        
        # Période d'analyse
        nb_jours = st.slider("Nombre de jours à analyser", 7, 90, 30)
        
        # Générer les données historiques
        coords = app.villes_allemandes[ville_hist]
        df_hist = app.generer_donnees_historiques(ville_hist, coords, jours=nb_jours)
        
        # Graphique temporel principal
        st.subheader(f"Évolution sur {nb_jours} jours - {ville_hist}")
        
        fig_timeline = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Prix de l\'électricité', 'Conditions météorologiques'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Prix
        fig_timeline.add_trace(
            go.Scatter(
                x=df_hist['Date'],
                y=df_hist['Prix'],
                mode='lines+markers',
                name='Prix (EUR/MWh)',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        # Météo
        fig_timeline.add_trace(
            go.Scatter(x=df_hist['Date'], y=df_hist['Température'], 
                      name='Température (°C)', line=dict(color='red')),
            row=2, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(x=df_hist['Date'], y=df_hist['Vent'], 
                      name='Vent (km/h)', line=dict(color='green')),
            row=2, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(x=df_hist['Date'], y=df_hist['Nuages'], 
                      name='Nuages (%)', line=dict(color='gray', dash='dot')),
            row=2, col=1
        )
        
        fig_timeline.update_xaxes(title_text="Date", row=2, col=1)
        fig_timeline.update_yaxes(title_text="EUR/MWh", row=1, col=1)
        fig_timeline.update_yaxes(title_text="Valeur", row=2, col=1)
        fig_timeline.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Statistiques sur la période
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric(
                "Prix moyen sur la période",
                f"{df_hist['Prix'].mean():.2f} €/MWh",
                delta=f"± {df_hist['Prix'].std():.2f}"
            )
        
        with col_stat2:
            st.metric(
                "Prix minimum",
                f"{df_hist['Prix'].min():.2f} €/MWh",
                delta=f"{df_hist.loc[df_hist['Prix'].idxmin(), 'Date']}"
            )
        
        with col_stat3:
            st.metric(
                "Prix maximum",
                f"{df_hist['Prix'].max():.2f} €/MWh",
                delta=f"{df_hist.loc[df_hist['Prix'].idxmax(), 'Date']}"
            )
        
        # Distribution des prix
        st.subheader("📊 Distribution des prix")
        
        fig_dist = px.histogram(
            df_hist,
            x='Prix',
            nbins=20,
            title='Distribution des prix sur la période',
            labels={'Prix': 'Prix (EUR/MWh)', 'count': 'Fréquence'},
            color_discrete_sequence=['steelblue']
        )
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # MODE 4: Analyse approfondie
    else:
        st.header("🔍 Analyse approfondie")
        
        # Options d'analyse
        analyse_type = st.radio(
            "Type d'analyse",
            ["Corrélations détaillées", "Analyse multivariée", "Carte de chaleur"],
            horizontal=True
        )
        
        if analyse_type == "Corrélations détaillées":
            st.subheader("📈 Analyse des corrélations")
            
            # Calculer toutes les corrélations
            colonnes_num = ['Température (°C)', 'Vent (km/h)', 'Nuages (%)', 'Prix (EUR/MWh)']
            corr_matrix = df_actuel[colonnes_num].corr()
            
            # Afficher les corrélations avec le prix
            prix_corr = corr_matrix['Prix (EUR/MWh)'].drop('Prix (EUR/MWh)').sort_values()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_corr_bar = px.bar(
                    x=prix_corr.values,
                    y=prix_corr.index,
                    orientation='h',
                    title='Corrélation avec le prix de l\'électricité',
                    labels={'x': 'Coefficient de corrélation', 'y': 'Variable'},
                    color=prix_corr.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_corr_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_corr_bar, use_container_width=True)
            
            with col2:
                # Interprétations
                st.markdown("### 📊 Interprétations")
                
                for var, corr in prix_corr.items():
                    if corr < -0.5:
                        force = "forte négative"
                        emoji = "📉"
                        explication = "Prix diminue significativement"
                    elif corr < -0.2:
                        force = "modérée négative"
                        emoji = "↘️"
                        explication = "Prix diminue légèrement"
                    elif corr < 0.2:
                        force = "faible"
                        emoji = "↔️"
                        explication = "Peu d'impact"
                    elif corr < 0.5:
                        force = "modérée positive"
                        emoji = "↗️"
                        explication = "Prix augmente légèrement"
                    else:
                        force = "forte positive"
                        emoji = "📈"
                        explication = "Prix augmente significativement"
                    
                    st.info(f"""
                    **{emoji} {var}**: Corrélation {force}
                    - Coefficient: {corr:+.3f}
                    - Impact: {explication}
                    """)
        
        elif analyse_type == "Analyse multivariée":
            st.subheader("🎯 Analyse multivariée")
            
            # Graphique 3D
            fig_3d = px.scatter_3d(
                df_actuel,
                x='Température (°C)',
                y='Vent (km/h)',
                z='Prix (EUR/MWh)',
                color='Nuages (%)',
                size='Prix (EUR/MWh)',
                hover_data=['Ville'],
                title='Visualisation 3D: Température × Vent × Prix',
                labels={
                    'Température (°C)': 'Température',
                    'Vent (km/h)': 'Vent',
                    'Prix (EUR/MWh)': 'Prix'
                }
            )
            fig_3d.update_layout(height=700)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Pairplot
            st.subheader("📊 Matrice de nuages de points")
            
            from plotly.subplots import make_subplots
            
            variables = ['Température (°C)', 'Vent (km/h)', 'Nuages (%)']
            n_vars = len(variables)
            
            fig_pairs = make_subplots(
                rows=n_vars,
                cols=n_vars,
                subplot_titles=[f"{v1} vs {v2}" for v1 in variables for v2 in variables]
            )
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i == j:
                        # Histogramme sur la diagonale
                        fig_pairs.add_trace(
                            go.Histogram(x=df_actuel[var1], name=var1, showlegend=False),
                            row=i+1, col=j+1
                        )
                    else:
                        # Scatter plot
                        fig_pairs.add_trace(
                            go.Scatter(
                                x=df_actuel[var2],
                                y=df_actuel[var1],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=df_actuel['Prix (EUR/MWh)'],
                                    colorscale='Viridis',
                                    showscale=(i==0 and j==n_vars-1)
                                ),
                                showlegend=False
                            ),
                            row=i+1, col=j+1
                        )
            
            fig_pairs.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_pairs, use_container_width=True)
        
        else:  # Carte de chaleur
            st.subheader("🗺️ Carte de chaleur géographique")
            
            # Simuler une carte avec les positions géographiques
            import plotly.graph_objects as go
            
            latitudes = [coords['lat'] for coords in app.villes_allemandes.values()]
            longitudes = [coords['lon'] for coords in app.villes_allemandes.values()]
            
            fig_map = go.Figure(go.Scattergeo(
                lon=longitudes,
                lat=latitudes,
                text=df_actuel['Ville'],
                mode='markers+text',
                marker=dict(
                    size=df_actuel['Prix (EUR/MWh)'] * 2,
                    color=df_actuel['Prix (EUR/MWh)'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Prix<br>(EUR/MWh)")
                ),
                textposition='top center',
                textfont=dict(size=10, color='black')
            ))
            
            fig_map.update_geos(
                center=dict(lat=51, lon=10),
                projection_scale=15,
                showcountries=True
            )
            
            fig_map.update_layout(
                title='Prix de l\'électricité par ville (taille = prix)',
                height=600,
                geo=dict(
                    scope='europe',
                    center=dict(lat=51, lon=10)
                )
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Tableau complet
            st.subheader("📋 Tableau de données complet")
            st.dataframe(
                df_actuel.style.background_gradient(
                    subset=['Prix (EUR/MWh)'],
                    cmap='RdYlGn_r'
                ).format(precision=2),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>⚡ Application d'analyse météo et prix de l'électricité en Allemagne</p>
        <p>Données simulées à des fins de démonstration | Pour données réelles, utiliser l'API ENTSO-E</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
