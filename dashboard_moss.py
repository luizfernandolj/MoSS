import streamlit as st
import plotly.express as px
import numpy as np
import ast
from utils.simulators import DirichletSimulator, MVNSimulator, UniformSimulator

# ============================================================
# Função de plotagem com Plotly Express (Reutilizada)
# ============================================================
def plot_3d(X, y, title):
    n_classes = len(np.unique(y))
    y_str = y.astype(str)

    # Paleta de cores vibrante e contrastante
    color_discrete_map = {
        '0': '#FF1E00',  # vermelho vibrante
        '1': '#0088FF',  # azul intenso
        '2': '#00C853',  # verde forte
        '3': '#FFD600',  # amarelo ouro
        '4': '#AA00FF',  # roxo vivo
    }

    if n_classes == 2:
        # Histograma interativo
        fig = px.histogram(
            x=X[:, 0],
            color=y_str,
            nbins=30,
            barmode="overlay",
            histnorm="probability density",
            title=title,
            labels={"x": "Score Classe 0", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_traces(opacity=0.6)
        fig.update_layout(
            xaxis_title="Score Classe 0",
            yaxis_title="Frequência",
            legend_title="Classe",
            template="plotly_white"
        )
        return fig

    elif n_classes == 3:
        # Scatter 3D interativo
        fig = px.scatter_3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            color=y_str,
            opacity=0.8,
            title=title,
            labels={"x": "Score Classe 0", "y": "Score Classe 1", "z": "Score Classe 2", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_layout(
            legend_title="Classe",
            template="plotly_white",
            height=800,
            font=dict(size=14)
        )
        return fig

    else:
        # Scatter 2D interativo (para n > 3 ou projeção genérica)
        # Se for > 3, pegamos as duas primeiras dimensões apenas para visualização 2D
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=y_str,
            opacity=0.8,
            title=title,
            labels={"x": "Score 0", "y": "Score 1", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_layout(legend_title="Classe", template="plotly_white")
        return fig

# ============================================================
# Configuração do Streamlit
# ============================================================
st.set_page_config(layout="wide", page_title="MoSS Explorer")
st.title("🎛️ MoSS Variant Explorer")
st.markdown("Explore os parâmetros das variantes do algoritmo MoSS (m, n, alphas) e visualize os resultados.")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Parâmetros")

# 1. Seleção da Variante
variant = st.sidebar.selectbox(
    "Variante MoSS",
    options=["MoSS (Binário)", "MoSS_MN (Multivariate Normal)", "MoSS_Dir (Dirichlet)"]
)

# 2. Parâmetro n (Tamanho da amostra)
n_samples = st.sidebar.number_input("Número de amostras (n)", min_value=10, max_value=100000, value=500, step=100)

# 3. Lógica específica por variante
merging_factor = 0.1
alpha = None
n_classes = 2

try:
    if variant == "MoSS (Binário)":
        n_classes = 2 # Fixo para MoSS binário original
        
        # Alpha (proporção classe 1)
        alpha = st.sidebar.slider("Alpha (Proporção classe 1)", 0.0, 1.0, 0.5)
        
        # Merging Factor (m) - scalar only for basic MoSS
        merging_factor = st.sidebar.slider("Merging Factor (m)", 0.0, 1.0, 0.1, step=0.05)
        
        # Executar
        X, y = UniformSimulator()(n=n_samples, alpha=alpha, merging_factor=merging_factor)
        title = f"MoSS Binário (n={n_samples}, m={merging_factor}, alpha={alpha})"
        
    else: # MoSS_MN ou MoSS_Dir
        # Número de Classes
        n_classes = st.sidebar.number_input("Número de Classes", min_value=2, max_value=10, value=2)
        
        # Merging Factor (m)
        use_m_vector = st.sidebar.checkbox("Usar vetor para 'm'?", value=False)
        if use_m_vector:
            m_input = st.sidebar.text_input(f"Vetor 'm' ({n_classes} valores, sep. por vírgula)", value="0.5, " * n_classes)
            # Parse input
            try:
                merging_factor = [float(x.strip()) for x in m_input.split(',') if x.strip()]
                if len(merging_factor) != n_classes:
                    st.error(f"Erro: O vetor m deve ter {n_classes} valores.")
                    st.stop()
            except ValueError:
                st.error("Erro: Certifique-se de usar apenas números separados por vírgula para m.")
                st.stop()
        else:
            merging_factor = st.sidebar.slider("Merging Factor (m)", 0.0, 1.0, 0.1, step=0.05)

        # Alpha (Vetor de proporções)
        use_custom_alphas = st.sidebar.checkbox("Usar Alphas customizados?", value=False)
        if use_custom_alphas:
            default_alphas = [round(1/n_classes, 2)] * n_classes
            # Ajustar último para somar 1 se necessário, mas apenas para display inicial
            default_str = ", ".join(map(str, default_alphas))
            alpha_input = st.sidebar.text_input(f"Vetor Alpha ({n_classes} valores, sep. por vírgula)", value=default_str)
            
            try:
                alpha = [float(x.strip()) for x in alpha_input.split(',') if x.strip()]
                if len(alpha) != n_classes:
                    st.error(f"Erro: O vetor alpha deve ter {n_classes} valores.")
                    st.stop()
                if not np.isclose(sum(alpha), 1.0, atol=0.05):
                    st.warning(f"Atenção: A soma dos alphas é {sum(alpha):.2f}, idealmente deve ser 1.0. O algoritmo normalizará ou usará o resto.")
            except ValueError:
                st.error("Erro: Certifique-se de usar apenas números separados por vírgula para alpha.")
                st.stop()
        else:
            # A prevalência é sempre explícita: o simulador toma dela o número
            # de classes, e não de um parâmetro separado que possa discordar.
            alpha = [1 / n_classes] * n_classes

        # Executar
        if variant == "MoSS_MN (Multivariate Normal)":
            X, y = MVNSimulator()(n=n_samples, alpha=alpha, merging_factor=merging_factor)
            title = f"MoSS MN (n={n_samples}, m={merging_factor})"
        else:
            X, y = DirichletSimulator()(n=n_samples, alpha=alpha, merging_factor=merging_factor)
            title = f"MoSS Dirichlet (n={n_samples}, m={merging_factor})"

    # ============================================================
    # Visualização
    # ============================================================
    st.subheader(title)
    
    # Debug info
    with st.expander("Ver Detalhes dos Dados"):
        st.write("Shape X:", X.shape)
        st.write("Shape y:", y.shape)
        st.write("Distribuição de Classes:", np.unique(y, return_counts=True))
        if alpha is not None:
            st.write("Alpha Usado:", alpha)
    
    fig = plot_3d(X, y, title)
    st.plotly_chart(fig, width='stretch')

except Exception as e:
    st.error(f"Ocorreu um erro durante a execução: {e}")
    # st.exception(e) # Uncomment to see full traceback
