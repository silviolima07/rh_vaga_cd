import numpy as np
from joblib import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pybase64
import sklearn
import tempfile
from time import sleep

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return pybase64.b64encode(image_file.read()).decode('utf-8') 
    


col5, col6 = st.columns(2)

with col5:

    html_page_title = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:blue'>Classificador de Candidatos</p>
</div>
"""               
    st.markdown(html_page_title, unsafe_allow_html=True)

with col6:

    job2 = Image.open('img/job.png')
    st.image(job2, use_container_width=False)

# Carregando o modelo Random Forest
def load_modelo():
    modelo = load("pipeline_rfc_cientista_dados.pkl")
    st.success("Modelo carregado com sucesso.")
    return modelo

# Mapeamento de proficiência
PROFICIENCIA_ORDER = ["Baixa", "Média", "Alta", "Avançada"]
PROFICIENCIA_MAP = {
    "Baixa": 0,
    "Média": 1,
    "Alta": 2,
    "Avançada": 3
}

# Função para gerar gráfico de importância
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def center_img(image, size):
    # Centralizar imagem da cloud
    img = Image.open(f'img/{image}.png')
    image_path="image.png"
    img.save(image_path)
    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Usando HTML para centralizar a imagem
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64_image}" alt="Imagem" style="width: {size}%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
        )


def show_feature_importance():
    preprocessor = modelo.named_steps['preprocessor']
    clf = modelo.named_steps['classifier']

    # Obtém nomes das colunas
    cat_features = preprocessor.transformers_[0][2]
    #num_features = preprocessor.transformers_[1][2]
    num_features = 'experiencia'
 
    
    #num_features2 = 'experiencia'
    feature_names = cat_features + [num_features]

    #print('Numerical Festure:', num_features)
    #print('Categorical Festure:', cat_features)
    
    # Importâncias
    importances = clf.feature_importances_
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    # Gráfico
    #st.markdown("### Importância das Variáveis")
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Feature'] = df['Feature'].astype(str)
    ax.barh(df["Feature"], df["Importance"], color="orange")
    ax.set_title("Ranking das Features - Random Forest")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    
    # Adicionar os valores na frente das barras
    for i, (value, name) in enumerate(zip(df['Importance'], df['Feature'])):
        plt.text(value + 0.01, i, f"{value * 100:.2f}%", va='center')

    st.pyplot(fig)
    
    
st.sidebar.markdown('### Observações: ')
st.sidebar.markdown('### Baixa:    Conheço um pouco,o básico')
st.sidebar.markdown('### Média:    Consigo seguir um tutorial')
st.sidebar.markdown('### Alta:     Consigo desenvolver programas e análises')
st.sidebar.markdown('### Avançada: Experiência em diversos projetos')

# Centralizar imagem da cloud    
#img = 'job'        
#center_img(img, 30)   

st.write(' ') 
st.markdown("### Responda:")
st.markdown("#### Quantos anos de experiência?")
exp = st.slider("Quantos anos de experiência?", 0, 10, 6, label_visibility='collapsed')

st.markdown("#### Informe o nivel de proficiência :")
col1, col2, col3 = st.columns(3)

with col1:
   st.markdown("#### 1) Excel")   
   excel = st.radio(
    "Excel",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility='collapsed' 
    )
   st.markdown("#### 4) Estatistica") 
   estatistica = st.radio(
    "estatistica",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'
    )
   
  
with col2:
    st.markdown("#### 2) Power BI") 
    power_bi = st.radio(
    "power_bi",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'
    )
    st.markdown("#### 5) Python") 
    python = st.radio(
    "python",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'   
    )
    st.markdown("#### 7) Inglês") 
    ingles = st.radio(
    "ingles",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'    
    )
    
with col3:    
    st.markdown("#### 3) Inteligência Artificial") 
    IA = st.radio(
    "IA",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'
    )
    st.markdown("#### 6) Natural Language Processing") 
    NLP = st.radio(
    "NLP",
    ['Baixa', 'Média', 'Alta', 'Avançada'],
    horizontal=True,
    label_visibility= 'collapsed'   
    )


dados = [exp, excel, power_bi, IA, estatistica, python, NLP, ingles]
colunas = ['experiencia', 'excel', 'power_bi','IA', 'estatistica', 'python', 'NLP', 'ingles']

features = pd.DataFrame([dados], columns=colunas)

st.markdown("### Perfil do candidato")
st.table(features)

modelo = load_modelo()

st.write("Modelo:", str(modelo[-1][1]))
#st.write("Modelo:", modelo[1].__class__.__name__)

if st.button("Classificar"):

    # Previsão do modelo
    pred = modelo.predict(features)
    probs = modelo.predict_proba(features)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Exibe previsão e probabilidades
        st.markdown("### Previsão")
    
        classe = 'Apto' if pred == 1 else 'Inapto'
    
        st.write('Classe:', classe.upper())
    
    with col2:    
        colunas_prob = ['Inapto', 'Apto']  # Ajustar a ordem se necessário
        df_probs = pd.DataFrame(probs, columns=colunas_prob)

        st.markdown("### Probabilidades")
        st.dataframe(df_probs.style.format("{:.2%}"))

    # Explicação do porque usar o 0 e não 1
    #st.write(modelo.classes_)
    #['apto', 'inapto'] 0 / apto  1 / inapto
    #modelo.predict_proba(features)[0][0] -> probabilidade de 'apto'
    #modelo.predict_proba(features)[0][1] -> probabilidade de 'inapto'
    
    
    # Gerar resultados (com emojis e formatação HTML)
    resultado = f"""
    <div style='font-size: 40px; font-weight: bold; text-align: center;'>
        {'😄 Candidato Aprovado 😊 ' if classe == 'Apto' else '☹️ Candidato Reprovado'}
    </div>
    """
    st.markdown(resultado, unsafe_allow_html=True)
    
    if classe == 'Inapto':
        st.markdown("#### Estude os tópicos considerados mais importantes na classificação.")
        show_feature_importance()
    else:
        html_page_subtitle = """
          <div style="background-color:black;padding=60px">
            <p style='text-align:center;font-size:30px;font-weight:bold; color:white'>Pode comemorar.</p>
          </div>
        """               
        st.markdown(html_page_subtitle, unsafe_allow_html=True)
        # Imagem da comemoracao
        img = 'chop'        
        center_img(img, 70)
        st.balloons()
        sleep(5)        
        st.balloons()        
        
    


