"""
=============================================================
MODSIM 2026 - Praktikum 5
Studi Kasus: Simulasi Proyek Pembangunan Gedung FITE 5 Lantai
Metode: PERT/CPM + Monte Carlo Simulation
=============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-sub {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid #bfdbfe;
        margin-bottom: 1.25rem;
        font-size: 0.92rem;
        color: #1e40af;
        line-height: 1.6;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        color: white;
        padding: 1.1rem 1rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(37,99,235,0.25);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }
    .metric-card p {
        margin: 0.25rem 0 0 0;
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stage-card {
        background: #f8fafc;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #2563eb;
        font-size: 0.88rem;
    }
    .critical-badge {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .rekomendasi-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.92rem;
        color: #064e3b;
        line-height: 1.7;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================

class TahapanProyek:
    """Memodelkan satu tahapan proyek dengan estimasi PERT dan faktor risiko."""

    def __init__(self, nama, params_pert, faktor_risiko=None, dependensi=None):
        self.nama = nama
        self.optimis    = params_pert['optimis']
        self.most_likely = params_pert['most_likely']
        self.pesimis    = params_pert['pesimis']
        self.faktor_risiko = faktor_risiko or {}
        self.dependensi    = dependensi or []

    def sampel_durasi(self, n_sim):
        """Sampling durasi dari distribusi PERT (via triangular) ditambah gangguan risiko."""
        durasi = np.random.triangular(
            self.optimis,
            self.most_likely,
            self.pesimis,
            n_sim
        )

        for _, params in self.faktor_risiko.items():
            if params['tipe'] == 'diskrit':
                terjadi = np.random.random(n_sim) < params['probabilitas']
                durasi = np.where(terjadi, durasi * (1 + params['dampak']), durasi)
            elif params['tipe'] == 'kontinu':
                faktor = np.random.normal(params['rata'], params['std'], n_sim)
                durasi = durasi / np.clip(faktor, 0.5, 1.5)

        return np.maximum(durasi, self.optimis * 0.5)


class SimulasiMonteCarlo:
    """Mengelola seluruh simulasi Monte Carlo untuk proyek konstruksi."""

    def __init__(self, konfigurasi, n_sim=10_000):
        self.konfigurasi = konfigurasi
        self.n_sim = n_sim
        self.tahapan: dict[str, TahapanProyek] = {}
        self.hasil = None
        self._inisialisasi_tahapan()

    def _inisialisasi_tahapan(self):
        for kode, cfg in self.konfigurasi.items():
            self.tahapan[kode] = TahapanProyek(
                nama=cfg['nama'],
                params_pert=cfg['pert'],
                faktor_risiko=cfg.get('risiko', {}),
                dependensi=cfg.get('dependensi', [])
            )

    def jalankan(self):
        """Forward pass Monte Carlo dengan dependensi antar tahapan."""
        durasi_sim = {k: t.sampel_durasi(self.n_sim) for k, t in self.tahapan.items()}

        es = {}   # Early Start
        ef = {}   # Early Finish

        # Urutan topologis sederhana: iterasi sampai semua selesai
        selesai = set()
        antrian = list(self.tahapan.keys())
        iterasi = 0
        while antrian and iterasi < len(antrian) * 2:
            iterasi += 1
            berikutnya = []
            for k in antrian:
                deps = self.tahapan[k].dependensi
                if all(d in selesai for d in deps):
                    if not deps:
                        es[k] = np.zeros(self.n_sim)
                    else:
                        es[k] = np.max(np.stack([ef[d] for d in deps]), axis=0)
                    ef[k] = es[k] + durasi_sim[k]
                    selesai.add(k)
                else:
                    berikutnya.append(k)
            antrian = berikutnya

        total_durasi = np.max(np.stack(list(ef.values())), axis=0)

        df = pd.DataFrame(durasi_sim)
        df['Total_Durasi'] = total_durasi
        for k in self.tahapan:
            df[f'{k}_ES'] = es[k]
            df[f'{k}_EF'] = ef[k]

        self.hasil = df
        self._ef = ef
        return df

    def analisis_critical_path(self):
        """Hitung probabilitas tiap tahapan ada di jalur kritis."""
        if self.hasil is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        total = self.hasil['Total_Durasi']
        data = []
        for k, t in self.tahapan.items():
            ef_k = self._ef[k]
            prob = np.mean(np.abs(ef_k - total) < 0.05)
            korelasi = self.hasil[k].corr(total)
            data.append({
                'Kode': k,
                'Tahapan': t.nama,
                'Probabilitas Kritis': prob,
                'Korelasi dgn Total': round(korelasi, 3),
                'Rata-rata Durasi (bln)': round(self.hasil[k].mean(), 2),
                'Std Dev': round(self.hasil[k].std(), 2),
            })
        return pd.DataFrame(data).set_index('Kode')

    def analisis_risiko(self):
        """Hitung kontribusi variansi tiap tahapan terhadap total proyek."""
        if self.hasil is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        total = self.hasil['Total_Durasi']
        var_total = total.var()
        data = []
        for k, t in self.tahapan.items():
            var_k = self.hasil[k].var()
            cov_k = self.hasil[k].cov(total)
            kontribusi = (cov_k / var_total) * 100
            data.append({
                'Kode': k,
                'Tahapan': t.nama,
                'Variansi': round(var_k, 4),
                'Std Dev': round(np.sqrt(var_k), 3),
                'Kontribusi (%)': round(kontribusi, 2),
            })
        return pd.DataFrame(data).set_index('Kode')


# ============================================================================
# 3. KONFIGURASI DEFAULT PROYEK GEDUNG FITE
# ============================================================================

KONFIGURASI_DEFAULT = {
    'A': {
        'nama': 'Perencanaan & Perizinan',
        'pert': {'optimis': 1.0, 'most_likely': 1.5, 'pesimis': 3.0},
        'risiko': {
            'birokrasi': {'tipe': 'diskrit', 'probabilitas': 0.25, 'dampak': 0.30}
        },
        'dependensi': []
    },
    'B': {
        'nama': 'Persiapan Lahan & Mobilisasi',
        'pert': {'optimis': 0.5, 'most_likely': 1.0, 'pesimis': 2.0},
        'risiko': {
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.30, 'dampak': 0.20}
        },
        'dependensi': ['A']
    },
    'C': {
        'nama': 'Pondasi & Struktur Bawah',
        'pert': {'optimis': 2.0, 'most_likely': 3.0, 'pesimis': 5.0},
        'risiko': {
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.35, 'dampak': 0.25},
            'material': {'tipe': 'diskrit', 'probabilitas': 0.20, 'dampak': 0.15}
        },
        'dependensi': ['B']
    },
    'D': {
        'nama': 'Struktur Lantai 1-2',
        'pert': {'optimis': 2.0, 'most_likely': 3.0, 'pesimis': 4.5},
        'risiko': {
            'produktivitas': {'tipe': 'kontinu', 'rata': 1.0, 'std': 0.15}
        },
        'dependensi': ['C']
    },
    'E': {
        'nama': 'Struktur Lantai 3-4',
        'pert': {'optimis': 2.0, 'most_likely': 3.0, 'pesimis': 4.5},
        'risiko': {
            'produktivitas': {'tipe': 'kontinu', 'rata': 1.0, 'std': 0.15},
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.30, 'dampak': 0.20}
        },
        'dependensi': ['D']
    },
    'F': {
        'nama': 'Struktur Lantai 5 & Atap',
        'pert': {'optimis': 1.5, 'most_likely': 2.0, 'pesimis': 3.5},
        'risiko': {
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.35, 'dampak': 0.25}
        },
        'dependensi': ['E']
    },
    'G': {
        'nama': 'Instalasi MEP',
        'pert': {'optimis': 1.5, 'most_likely': 2.5, 'pesimis': 4.0},
        'risiko': {
            'material_teknis': {'tipe': 'diskrit', 'probabilitas': 0.25, 'dampak': 0.20}
        },
        'dependensi': ['D']
    },
    'H': {
        'nama': 'Dinding & Fasad',
        'pert': {'optimis': 1.0, 'most_likely': 2.0, 'pesimis': 3.0},
        'risiko': {
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.25, 'dampak': 0.15}
        },
        'dependensi': ['F']
    },
    'I': {
        'nama': 'Finishing Interior Lt 1-2',
        'pert': {'optimis': 1.0, 'most_likely': 2.0, 'pesimis': 3.5},
        'risiko': {
            'desain_lab': {'tipe': 'diskrit', 'probabilitas': 0.20, 'dampak': 0.30},
            'produktivitas': {'tipe': 'kontinu', 'rata': 1.0, 'std': 0.20}
        },
        'dependensi': ['G', 'H']
    },
    'J': {
        'nama': 'Finishing Interior Lt 3-4',
        'pert': {'optimis': 1.0, 'most_likely': 2.0, 'pesimis': 3.5},
        'risiko': {
            'desain_lab': {'tipe': 'diskrit', 'probabilitas': 0.20, 'dampak': 0.30}
        },
        'dependensi': ['I']
    },
    'K': {
        'nama': 'Lab Komputer & Lab Elektro',
        'pert': {'optimis': 0.5, 'most_likely': 1.0, 'pesimis': 2.0},
        'risiko': {
            'material_teknis': {'tipe': 'diskrit', 'probabilitas': 0.30, 'dampak': 0.25}
        },
        'dependensi': ['J']
    },
    'L': {
        'nama': 'Lab Mobile, VR/AR & Lab Game',
        'pert': {'optimis': 0.5, 'most_likely': 1.0, 'pesimis': 2.0},
        'risiko': {
            'material_teknis': {'tipe': 'diskrit', 'probabilitas': 0.30, 'dampak': 0.25}
        },
        'dependensi': ['J']
    },
    'M': {
        'nama': 'Ruang Dosen & Serbaguna',
        'pert': {'optimis': 0.5, 'most_likely': 0.8, 'pesimis': 1.5},
        'risiko': {},
        'dependensi': ['J']
    },
    'N': {
        'nama': 'Toilet & Utilitas',
        'pert': {'optimis': 0.3, 'most_likely': 0.5, 'pesimis': 1.0},
        'risiko': {},
        'dependensi': ['I']
    },
    'O': {
        'nama': 'Finishing Eksterior & Lansekap',
        'pert': {'optimis': 0.5, 'most_likely': 1.0, 'pesimis': 2.0},
        'risiko': {
            'cuaca': {'tipe': 'diskrit', 'probabilitas': 0.30, 'dampak': 0.20}
        },
        'dependensi': ['H', 'N']
    },
    'P': {
        'nama': 'Pengujian, Inspeksi & Serah Terima',
        'pert': {'optimis': 0.5, 'most_likely': 1.0, 'pesimis': 1.5},
        'risiko': {},
        'dependensi': ['K', 'L', 'M', 'O']
    },
}

DEADLINES_BULAN = [16, 20, 24]


# ============================================================================
# 4. FUNGSI VISUALISASI
# ============================================================================

WARNA = {
    'biru': '#2563eb', 'navy': '#1e3a5f', 'merah': '#dc2626',
    'hijau': '#16a34a', 'oranye': '#ea580c', 'kuning': '#ca8a04',
    'abu': '#64748b', 'ungu': '#7c3aed',
}


def plot_distribusi(hasil):
    total = hasil['Total_Durasi']
    mean_v = total.mean()
    med_v  = np.median(total)
    ci80   = np.percentile(total, [10, 90])
    ci95   = np.percentile(total, [2.5, 97.5])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total, nbinsx=60,
        name='Distribusi Durasi',
        marker_color=WARNA['biru'], opacity=0.75,
        histnorm='probability density'
    ))
    fig.add_vline(x=mean_v, line_dash='dash', line_color=WARNA['merah'],
                  annotation_text=f'Rata-rata: {mean_v:.1f} bln', annotation_position='top right')
    fig.add_vline(x=med_v, line_dash='dot', line_color=WARNA['hijau'],
                  annotation_text=f'Median: {med_v:.1f} bln')
    fig.add_vrect(x0=ci80[0], x1=ci80[1], fillcolor='yellow', opacity=0.15,
                  annotation_text='80% CI', line_width=0)
    fig.add_vrect(x0=ci95[0], x1=ci95[1], fillcolor='orange', opacity=0.08,
                  annotation_text='95% CI', line_width=0)
    for dl, col in zip(DEADLINES_BULAN, [WARNA['hijau'], WARNA['oranye'], WARNA['merah']]):
        prob = np.mean(total <= dl)
        fig.add_vline(x=dl, line_color=col, line_width=1.5,
                      annotation_text=f'{dl} bln ({prob:.0%})')
    fig.update_layout(
        title='Distribusi Durasi Total Proyek Gedung FITE',
        xaxis_title='Durasi Proyek (Bulan)', yaxis_title='Densitas Probabilitas',
        height=460, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig, {'mean': mean_v, 'median': med_v, 'std': total.std(),
                 'min': total.min(), 'max': total.max(), 'ci80': ci80, 'ci95': ci95}


def plot_cdf(hasil):
    total = hasil['Total_Durasi']
    x = np.sort(total)
    y = np.arange(1, len(x)+1) / len(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y*100, mode='lines',
                             name='P(Selesai)',
                             line=dict(color=WARNA['navy'], width=3),
                             fill='tozeroy',
                             fillcolor='rgba(37,99,235,0.10)'))
    for lv, col in [(50, WARNA['merah']), (80, WARNA['oranye']), (95, WARNA['hijau'])]:
        deadline_lv = np.percentile(total, lv)
        fig.add_hline(y=lv, line_dash='dash', line_color=col,
                      annotation_text=f'{lv}%', annotation_position='right')
        fig.add_vline(x=deadline_lv, line_dash='dot', line_color=col,
                      annotation_text=f'{deadline_lv:.1f} bln')

    for dl, col in zip(DEADLINES_BULAN, [WARNA['hijau'], WARNA['oranye'], WARNA['merah']]):
        prob = np.mean(total <= dl) * 100
        fig.add_trace(go.Scatter(
            x=[dl], y=[prob], mode='markers+text',
            marker=dict(size=13, color=col, line=dict(width=2, color='white')),
            text=[f'{prob:.1f}%'], textposition='top center',
            showlegend=False
        ))

    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek (CDF)',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Penyelesaian (%)',
        yaxis_range=[-3, 103],
        height=460, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def plot_critical_path(cp_df):
    cp_sorted = cp_df.sort_values('Probabilitas Kritis', ascending=True)
    colors = [WARNA['merah'] if p > 0.6 else
              (WARNA['oranye'] if p > 0.3 else WARNA['biru'])
              for p in cp_sorted['Probabilitas Kritis']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cp_sorted['Tahapan'],
        x=cp_sorted['Probabilitas Kritis'],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in cp_sorted['Probabilitas Kritis']],
        textposition='auto'
    ))
    fig.add_vline(x=0.5, line_dash='dot', line_color='gray',
                  annotation_text='50%', annotation_position='top')
    fig.add_vline(x=0.7, line_dash='dot', line_color='orange',
                  annotation_text='70%', annotation_position='top')
    fig.update_layout(
        title='Probabilitas Tiap Tahapan Menjadi Jalur Kritis',
        xaxis_title='Probabilitas Jalur Kritis', xaxis_range=[0, 1.05],
        height=500, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def plot_boxplot(hasil, tahapan):
    kode_list = list(tahapan.keys())
    fig = go.Figure()
    palette = px.colors.qualitative.Pastel
    for i, k in enumerate(kode_list):
        fig.add_trace(go.Box(
            y=hasil[k], name=k,
            boxmean='sd',
            marker_color=palette[i % len(palette)],
            boxpoints='outliers',
            hovertext=tahapan[k].nama,
        ))
    fig.update_layout(
        title='Distribusi Durasi per Tahapan (Box Plot)',
        yaxis_title='Durasi (Bulan)',
        height=480, template='plotly_white', showlegend=False,
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def plot_gantt(hasil, tahapan):
    """Gantt chart berdasarkan nilai rata-rata simulasi."""
    data = []
    for k, t in tahapan.items():
        data.append({
            'Kode': k,
            'Tahapan': t.nama,
            'Mulai': hasil[f'{k}_ES'].mean(),
            'Selesai': hasil[f'{k}_EF'].mean(),
        })
    df = pd.DataFrame(data)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Selesai'] - row['Mulai']],
            y=[f"{row['Kode']}: {row['Tahapan'][:28]}"],
            base=row['Mulai'],
            orientation='h',
            marker_color=WARNA['biru'],
            opacity=0.80,
            text=row['Kode'],
            textposition='inside',
            showlegend=False,
            hovertemplate=f"<b>{row['Tahapan']}</b><br>Mulai: {row['Mulai']:.1f} bln<br>Selesai: {row['Selesai']:.1f} bln<extra></extra>"
        ))

    fig.update_layout(
        title='Gantt Chart Rata-rata Durasi Tahapan',
        xaxis_title='Bulan ke-',
        barmode='overlay',
        height=560, template='plotly_white',
        font=dict(family='Plus Jakarta Sans'),
        yaxis=dict(autorange='reversed')
    )
    return fig


def plot_kontribusi_risiko(risk_df):
    risk_sorted = risk_df.sort_values('Kontribusi (%)', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=risk_sorted['Tahapan'],
        y=risk_sorted['Kontribusi (%)'],
        marker_color=px.colors.qualitative.Set3,
        text=[f"{v:.1f}%" for v in risk_sorted['Kontribusi (%)']],
        textposition='auto'
    ))
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan terhadap Variabilitas Total',
        yaxis_title='Kontribusi Variansi (%)',
        xaxis_tickangle=-35,
        height=420, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def plot_heatmap_korelasi(hasil, tahapan):
    kode_list = list(tahapan.keys())
    corr = hasil[kode_list].corr()
    labels = [f"{k}: {tahapan[k].nama[:15]}" for k in kode_list]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={'size': 8},
    ))
    fig.update_layout(
        title='Matriks Korelasi Antar Tahapan',
        height=540, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def plot_skenario_resource(hasil):
    skenario = {
        'Baseline': 1.00,
        'Pekerja Khusus\n(+20% speed)': 0.83,
        'Alat Berat\n(+15% speed)': 0.87,
        'Insinyur Tambahan\n(+25% speed)': 0.80,
        'Kombinasi\n(+40% speed)': 0.71,
    }
    total = hasil['Total_Durasi']
    means = [total.mean() * f for f in skenario.values()]
    stds  = [total.std()  * f for f in skenario.values()]
    p20   = [np.mean(total * f <= 20) * 100 for f in skenario.values()]
    colors = [WARNA['abu']] + [WARNA['hijau']] * 4

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(skenario.keys()), y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=colors, opacity=0.85,
        text=[f'{m:.1f} bln<br>P20={p:.0f}%' for m, p in zip(means, p20)],
        textposition='outside'
    ))
    fig.add_hline(y=20, line_dash='dash', line_color=WARNA['merah'],
                  annotation_text='Target 20 bulan')
    fig.add_hline(y=16, line_dash='dash', line_color=WARNA['oranye'],
                  annotation_text='Target 16 bulan')
    fig.update_layout(
        title='Analisis Skenario Penambahan Resource',
        yaxis_title='Rata-rata Durasi Proyek (Bulan)',
        height=440, template='plotly_white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


# ============================================================================
# 5. FUNGSI UTAMA STREAMLIT
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo — Pembangunan Gedung FITE 5 Lantai</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Pemodelan & Estimasi Waktu Proyek Berbasis PERT/CPM + Monte Carlo | MODSIM 2026 – Praktikum 5</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Proyek pembangunan Gedung FITE 5 lantai mencakup ruang kelas, lab komputer, lab elektro, lab mobile, lab VR/AR, 
    lab game, ruang dosen, toilet, dan ruang serbaguna. Simulasi ini menjawab pertanyaan: berapa lama total proyek? 
    Tahapan mana yang paling kritis? Berapa probabilitas selesai tepat deadline? Bagaimana dampak penambahan resource?
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")

    n_sim = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:', 1000, 50000, 10000, 1000,
        help='Semakin banyak iterasi → hasil lebih akurat tapi lebih lama.'
    )

    st.sidebar.markdown("### 📋 Parameter Tiap Tahapan (bulan)")
    cfg = {k: {**v} for k, v in KONFIGURASI_DEFAULT.items()}   # shallow copy

    for kode, data in cfg.items():
        with st.sidebar.expander(f"**{kode}** — {data['nama']}", expanded=False):
            o = st.number_input('Optimis (bln)',  0.1, 24.0, float(data['pert']['optimis']),    0.1, key=f"o_{kode}")
            m = st.number_input('Most Likely (bln)', 0.1, 24.0, float(data['pert']['most_likely']), 0.1, key=f"m_{kode}")
            p = st.number_input('Pesimis (bln)', 0.1, 24.0, float(data['pert']['pesimis']),  0.1, key=f"p_{kode}")
            cfg[kode]['pert'] = {'optimis': o, 'most_likely': m, 'pesimis': p}

    run_btn = st.sidebar.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.8rem;color:#666;">
    <b>Keterangan:</b><br>
    • Optimis: Durasi terbaik yang mungkin<br>
    • Most Likely: Durasi paling realistis<br>
    • Pesimis: Durasi terburuk yang mungkin<br>
    • CI: Confidence Interval<br>
    • MEP: Mechanical, Electrical, Plumbing
    </div>
    """, unsafe_allow_html=True)

    # ── Session State ────────────────────────────────────────────────────
    if 'hasil'     not in st.session_state: st.session_state.hasil     = None
    if 'simulator' not in st.session_state: st.session_state.simulator = None

    if run_btn:
        with st.spinner('⏳ Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            sim = SimulasiMonteCarlo(cfg, n_sim)
            hasil = sim.jalankan()
            st.session_state.hasil     = hasil
            st.session_state.simulator = sim
        st.success(f'✅ Simulasi selesai! {n_sim:,} iterasi berhasil dijalankan.')

    # ── Tampilkan hasil ──────────────────────────────────────────────────
    if st.session_state.hasil is not None:
        hasil = st.session_state.hasil
        sim   = st.session_state.simulator
        total = hasil['Total_Durasi']

        # ── Metric cards ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Statistik Utama Proyek</div>', unsafe_allow_html=True)
        mean_v = total.mean()
        med_v  = np.median(total)
        ci80   = np.percentile(total, [10, 90])
        ci95   = np.percentile(total, [2.5, 97.5])
        std_v  = total.std()

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label in zip(
            [c1, c2, c3, c4, c5],
            [f"{mean_v:.1f} bln", f"{med_v:.1f} bln", f"±{std_v:.1f} bln",
             f"{ci80[0]:.1f}–{ci80[1]:.1f}", f"{ci95[0]:.1f}–{ci95[1]:.1f}"],
            ["Rata-rata Durasi", "Median Durasi", "Std Deviasi", "80% CI (bulan)", "95% CI (bulan)"]
        ):
            col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Probabilitas Deadline ─────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Probabilitas Penyelesaian per Deadline</div>', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        for col, dl, col_label in zip([dc1, dc2, dc3], DEADLINES_BULAN, ['🟢', '🟡', '🔴']):
            prob = np.mean(total <= dl)
            label_status = "Sangat Mungkin" if prob >= 0.80 else ("Berisiko" if prob >= 0.40 else "Sangat Berisiko")
            col.metric(
                label=f"{col_label} Target {dl} Bulan",
                value=f"{prob:.1%}",
                delta=label_status
            )

        # ── Tab Visualisasi ───────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Visualisasi Hasil Simulasi</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Distribusi Durasi",
            "🎯 Kurva CDF",
            "🗂️ Gantt & Boxplot",
            "🔴 Critical Path",
            "⚡ Risiko & Resource"
        ])

        with tab1:
            fig_dist, stats_d = plot_distribusi(hasil)
            st.plotly_chart(fig_dist, use_container_width=True)
            with st.expander("📋 Detail Statistik Distribusi"):
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""
                    **Statistik Deskriptif:**
                    - Rata-rata : **{stats_d['mean']:.2f} bulan**
                    - Median    : **{stats_d['median']:.2f} bulan**
                    - Std Dev   : **{stats_d['std']:.2f} bulan**
                    - Minimum   : **{stats_d['min']:.2f} bulan**
                    - Maksimum  : **{stats_d['max']:.2f} bulan**
                    """)
                with sc2:
                    st.markdown(f"""
                    **Confidence Intervals:**
                    - 80% CI : **[{stats_d['ci80'][0]:.1f}, {stats_d['ci80'][1]:.1f}] bulan**
                    - 95% CI : **[{stats_d['ci95'][0]:.1f}, {stats_d['ci95'][1]:.1f}] bulan**
                    
                    **Persentil Penting:**
                    - P50  : {np.percentile(total, 50):.1f} bulan
                    - P80  : {np.percentile(total, 80):.1f} bulan
                    - P95  : {np.percentile(total, 95):.1f} bulan
                    """)

        with tab2:
            fig_cdf = plot_cdf(hasil)
            st.plotly_chart(fig_cdf, use_container_width=True)
            with st.expander("📅 Tabel Probabilitas Deadline"):
                deadline_list = list(range(12, 31, 1))
                rows = []
                for dl in deadline_list:
                    prob = np.mean(total <= dl)
                    rows.append({'Deadline (bulan)': dl,
                                 'Probabilitas Selesai': f'{prob:.1%}',
                                 'Probabilitas Terlambat': f'{1-prob:.1%}'})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab3:
            gc1, gc2 = st.columns([3, 2])
            with gc1:
                st.plotly_chart(plot_gantt(hasil, sim.tahapan), use_container_width=True)
            with gc2:
                st.plotly_chart(plot_boxplot(hasil, sim.tahapan), use_container_width=True)

        with tab4:
            cp_df = sim.analisis_critical_path()
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(plot_critical_path(cp_df), use_container_width=True)
            with cc2:
                st.plotly_chart(plot_heatmap_korelasi(hasil, sim.tahapan), use_container_width=True)
            with st.expander("🔍 Detail Tabel Critical Path"):
                st.dataframe(
                    cp_df.style.background_gradient(subset=['Probabilitas Kritis'], cmap='Reds'),
                    use_container_width=True
                )

        with tab5:
            rc1, rc2 = st.columns(2)
            risk_df = sim.analisis_risiko()
            with rc1:
                st.plotly_chart(plot_kontribusi_risiko(risk_df), use_container_width=True)
            with rc2:
                st.plotly_chart(plot_skenario_resource(hasil), use_container_width=True)
            with st.expander("📋 Detail Tabel Kontribusi Risiko"):
                st.dataframe(
                    risk_df.style.background_gradient(subset=['Kontribusi (%)'], cmap='Oranges'),
                    use_container_width=True
                )

        # ── Analisis Deadline & Rekomendasi ───────────────────────────
        st.markdown('<div class="section-header">🎯 Analisis Deadline & Rekomendasi</div>', unsafe_allow_html=True)
        ra1, ra2 = st.columns(2)

        with ra1:
            target = st.number_input("Masukkan deadline target (bulan):", 10, 36, 24, 1)
            prob_target = np.mean(total <= target)
            risk_days   = max(0.0, float(np.percentile(total, 95) - target))
            st.metric(
                label=f"Probabilitas selesai dalam {target} bulan",
                value=f"{prob_target:.1%}",
                delta=f"Potensi keterlambatan: {risk_days:.1f} bln" if risk_days > 0 else "Aman ✅",
                delta_color="inverse"
            )

        with ra2:
            safety_buf = float(np.percentile(total, 80) - mean_v)
            conting    = float(np.percentile(total, 95) - mean_v)
            st.markdown(f"""
            <div class="rekomendasi-box">
            <b>🏗️ Rekomendasi Manajemen Proyek:</b><br><br>
            • <b>Safety Buffer</b> (80% CI): <b>{safety_buf:.1f} bulan</b><br>
            • <b>Contingency Reserve</b> (95% CI): <b>{conting:.1f} bulan</b><br><br>
            • <b>Jadwal yang direkomendasikan:</b><br>
              &nbsp;&nbsp;{mean_v:.1f} + {safety_buf:.1f} = <b>{mean_v + safety_buf:.1f} bulan</b><br><br>
            • Fokus percepatan pada tahapan: <b>Pondasi, Struktur, & Finishing Interior</b>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabel Statistik Lengkap ────────────────────────────────────
        st.markdown('<div class="section-header">📋 Statistik Lengkap per Tahapan</div>', unsafe_allow_html=True)
        rows_stat = []
        for k, t in sim.tahapan.items():
            d = hasil[k]
            rows_stat.append({
                'Kode': k, 'Tahapan': t.nama,
                'Optimis (input)': t.optimis,
                'Most Likely (input)': t.most_likely,
                'Pesimis (input)': t.pesimis,
                'Rata-rata Sim.': round(d.mean(), 2),
                'Std Dev': round(d.std(), 2),
                'P25': round(np.percentile(d, 25), 2),
                'P50': round(np.percentile(d, 50), 2),
                'P75': round(np.percentile(d, 75), 2),
            })
        st.dataframe(pd.DataFrame(rows_stat).set_index('Kode'), use_container_width=True)

        # ── Info Teknis ───────────────────────────────────────────────
        with st.expander("ℹ️ Informasi Teknis Simulasi"):
            st.markdown(f"""
            **Parameter Simulasi:**
            - Jumlah iterasi : {n_sim:,}
            - Jumlah tahapan : {len(sim.tahapan)}
            - Distribusi sampling : Triangular (PERT)
            - Faktor risiko : Diskrit (Bernoulli) & Kontinu (Normal)

            **Faktor Ketidakpastian yang Dimodelkan:**
            - 🌧️ Cuaca buruk → penundaan diskrit (15–30%)
            - 📦 Keterlambatan material teknis → penundaan diskrit (15–25%)
            - 🏛️ Perubahan desain laboratorium → penundaan diskrit (25–35%)
            - 👷 Produktivitas pekerja → gangguan kontinu (Normal)
            - 📜 Birokrasi perizinan → penundaan diskrit (30%)
            """)

    else:
        # Placeholder sebelum simulasi dijalankan
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; background:#f8fafc;
                    border-radius:16px; border:2px dashed #cbd5e1;">
            <h3 style="color:#1e3a5f;">🚀 Siap Memulai Simulasi?</h3>
            <p style="color:#64748b;">Atur parameter PERT di sidebar kiri, lalu klik <b>"Jalankan Simulasi"</b>.</p>
            <p style="color:#64748b;">Hasil analisis lengkap akan muncul di sini.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 Preview Konfigurasi Tahapan</div>', unsafe_allow_html=True)
        for k, v in KONFIGURASI_DEFAULT.items():
            p = v['pert']
            dep = ', '.join(v['dependensi']) if v['dependensi'] else '—'
            st.markdown(f"""
            <div class="stage-card">
            <b>{k}</b> — {v['nama']} &nbsp;|&nbsp;
            O={p['optimis']} · M={p['most_likely']} · P={p['pesimis']} bln &nbsp;|&nbsp;
            Dependensi: {dep}
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
    <b>MODSIM 2026 – Praktikum 5 | Studi Kasus Pembangunan Gedung FITE 5 Lantai</b><br>
    ⚠️ Hasil simulasi merupakan estimasi probabilistik, bukan prediksi pasti.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()