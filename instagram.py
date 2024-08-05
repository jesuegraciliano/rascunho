import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    tool,
    SerperDevTool,
    ScrapeWebsiteTool,
    WebsiteSearchTool,
    DirectoryReadTool,
    FileReadTool,
    PDFSearchTool,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# Função para ler a chave da API Groc
def get_groq_api_key():
    with open("API_LLM_Groq.txt", "r") as file:
        return file.read().strip()


# Função para ler a chave da API OpenAI
def get_openai_api_key():
    with open("API_LLM_OpenAI.txt", "r") as file:
        return file.read().strip()
    # Função para ler a chave da API Serper


def get_serper_api_key():
    with open("API_Serper.txt", "r") as file:
        return file.read().strip()
    groq_api_key = get_groq_api_key()


openai_api_key = get_openai_api_key()
serper_api_key = get_serper_api_key()
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key
gpt3_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
llama3_70b = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
gpt4o_mini_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
# Ferramenta para busca no google
search_tool = SerperDevTool()

# Ferramenta para raspagem de sites
scrape_tool = ScrapeWebsiteTool()

# Especificação de apenas um site para raspagem
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://www.cultura.sc.gov.br/espacos/mhsc"
)

# Ferramenta para leitura de diretório
directory_read_tool = DirectoryReadTool(directory="./informacoes_relevantes")

# Ferramenta para leitura de arquivo. Carrega toda a informação do arquivo para contexto e futura análise
file_read_tool = FileReadTool()

# Ferramenta para RAG em arquivos PDF. Específica para fazer buscas e extrair partes relevantes em arquivos PDF.
pdf_search = PDFSearchTool(pdf="Palacio_Cruz_Souza.pdf")
controles = """
\n<controle>\n
NÍVEIS DE CONTROLE:\n
1. Entonação: Informal.\n
2. Foco de Tópico: Você deve gerar o conteúdo sempre com alto foco no tema e público-alvo definidos.\n
3. Idioma: Gere o conteúdo sempre em Português do Brasil.\n
4. Controle de Sentimento: Use um tom amigável e encorajador, com alguns adjetivos positivos e exclamações moderadas. Evite superlativos como: inovador, revolucionário e etc\n
5. Nível Originalidade: 10, onde 1 é pouco original e 10 é muito original. Em hipótese alguma copie posts já publicados.\n
</controle>\n
"""
planejador = Agent(
    role="Planejador de Conteúdo",
    goal="Planejar conteúdo envolvente e factualmente preciso sobre {tema}"
    "para {publico_alvo}",
    backstory="Você está trabalhando no planejamento de um post no"
    "estilo carrossel para o Instagram sobre o tema: {tema}"
    "direcionado para o público: {publico_alvo}. Você coleta as últimas"
    "tendências, considera informações específicas que são fornecidas"
    "e busca as estratégias de postagens mais adequadas"
    "e relevantes sobre {tema}, visando criar conteúdos cativantes"
    "para {publico_alvo}. Você também busca no Instragram"
    "perfis de destaque relacionados ao tema {tema} para inspirar"
    "seu planejamento. Seu trabalho é a base para que o Redator de Conteúdo"
    "escreva um post no estilo carrossel sobre {tema}.",
    verbose=True,
    tools=[search_tool, scrape_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, pdf_search, search_tool, scrape_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, pdf_search, docs_scrape_tool],
    allow_delegation=False,
    llm=gpt4o_mini_llm,
)
redator = Agent(
    role="Redator de Conteúdo",
    goal="Escrever, em português do Brasil, um post no estilo carrossel"
    "para o Instagram sobre o tema: {tema} visnado atingir {publico_alvo}",
    backstory="Você está trabalhando na escrita "
    "de um post no estilo carrossel sobre o tema: {tema}"
    "para {publico_alvo}. Você se destaca em criar narrativas que"
    "ressoam com {publico_alvo} nas mídias sociais."
    "Você baseia sua escrita no trabalho do "
    "Planejador de Conteúdo, que fornece um esboço "
    "e contexto relevante sobre {tema}. "
    "Você segue os principais objetivos e "
    "a direção do esboço, conforme fornecido pelo Planejador de Conteúdo. "
    "Você também utiliza as tendências relevantes encontradas"
    "pelo Pesquisador de Conteúdo para deixar seu post atraentes e atualizado."
    "Você deve levar em consideração as definições de controles em <controle></controle>."
    "Seu trabalho é a base para que o Designer Gráfico encontre as melhores"
    "imagens para o post no estilo carrossel."
    "{controles}",
    verbose=True,
    tools=[search_tool, scrape_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, pdf_search, search_tool, scrape_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, docs_scrape_tool],
    #    tools=[directory_read_tool, file_read_tool, pdf_search, docs_scrape_tool],
    allow_delegation=False,
    llm=gpt4o_mini_llm,
)
designer = Agent(
    role="Designer Gráfico",
    goal="Buscar as imagens mais incríveis para anúncios no Instagram que"
    "capturam emoções e transmitem uma mensagem convincente.",
    backstory="Como Designer gráfico Sênior em uma agência de marketing digital"
    "líder, você é especialista em encontrar imagens incríveis"
    "que inspiram e envolvem o público: {publico_alvo}. Agora você está trabalhando na criação"
    "de um post no estilo carrossel para um cliente super importante"
    "e precisa encontrar as imagens mais incrível. Você utiliza"
    "o conteúdo do post gerado pelo Redator de Conteúdo"
    "como base para buscar as imagens.",
    verbose=True,
    tools=[search_tool, scrape_tool, docs_scrape_tool],
    allow_delegation=False,
    llm=gpt4o_mini_llm,
)
planejamento = Task(
    description=(
        "1. Priorize as últimas tendências, os principais players "
        "e as notícias relevantes sobre {tema}.\n"
        "2. Considere informações específicas sobre {tema} que são fornecidas em sua base"
        "3. Considere os interesses e pontos de desejo do"
        "público-alvo: {publico_alvo}.\n"
        "4. Desenvolva um esboço de conteúdo detalhado, incluindo "
        "pontos-chave, descritivos de imagens e um call to action.\n"
        "5. Inclua o número de slides que deve conter o carrossel"
        "dando o devido encadeamento entre um slide e outro.\n"
        "6. O conteúdo deve ser chamativo e incentivar os público-alvo"
        "a tomar uma ação, seja visitando o site, fazendo uma compra"
        "ou aprendendo mais sobre o tema.\n"
    ),
    expected_output="Um documento de plano de conteúdo abrangente "
    "com um esboço, análise de público, "
    "palavras-chave para compor os slides do"
    "carrossel e demais recursos.",
    agent=planejador,
)
redigir = Task(
    description=(
        "1. Utilize a saída da tarefa planejamento para elaborar um "
        "post no estilo carrossel atraente sobre {tema}.\n"
        "2. Incorpore as palavras-chave de forma natural.\n"
        "3. O encadeamento dos slides devem ser bem atrativos"
        "e envolventes.\n"
        "4. Certifique-se de que o post esteja atrativo para"
        "o público-alvo: {publico_alvo}.\n"
        "5. Revise para corrigir erros gramaticais e "
        "alinhamento com o estilo do tema: {tema}.\n"
        "6. Certifique-se de seguir o plano de conteúdo"
        "elaborado pelo agente planejador\n."
        "7. Sua resposta final DEVE ser um posts para Instagram"
        "que não apenas informe, mas também entusiasme e persuada o público.\n"
    ),
    expected_output="Um post no estilo carrossel bem escrito,"
    "pronto para o Designer Gráfico buscar as imagens necessárias"
    "para compor o post final.",
    agent=redator,
)
arte_grafica = Task(
    description=(
        "1. Use o post produzido pelo agente redator para buscar"
        "imagens que representem o conteúdo de cada slide do carrossel.\n"
        "2. Suas imagens devem ser acompanhadas de uma descrição detalhada.\n"
    ),
    expected_output="link para as imagens escolhidas para cada slide do carrossel que represente"
    "o conteúdo redigido pelo redator.",
    agent=designer,
)
revisao = Task(
    description=(
        "1. Revise o conteúdo e as imagens geradas para garantir que estejam"
        "sem erros de ortografia, consistentes com o tema: {tema}"
        "e estejam atrativas para o público-alvo {publico_alvo}.\n"
        "2. Verifique se o agente Redator de Conteúdo seguiu o plano de conteúdo"
        "elaborado pelo agente Planejador de Conteúdos e se o agente Designer Gráfico"
        "seguiu o conteúdo do post elaborado pelo Redator de Conteúdo para"
        "para definiar as imagens.\n"
    ),
    expected_output="Um post no estilo carrossel para o Instagram revisado,"
    "bem escrito, contendo o texto, a descrição e um link para cada imagem e pronto para publicação.\n",
    agent=diretor,
)
crew = Crew(
    agents=[planejador, redator, designer, diretor],
    tasks=[planejamento, redigir, arte_grafica, revisao],
    verbose=2,
)
entradas = {
    "tema": "Museu Cruz e Souza Florianópolis",
    "publico_alvo": "Pessoas interessadas em cultura",
    "controles": controles,
}
result = crew.kickoff(inputs=entradas)
from IPython.display import Markdown

Markdown(result)
