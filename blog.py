import os
from crewai_tools import tool
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Função para ler a chave de API do arquivo
def get_openai_api_key():
    with open('APIJESUE.txt', 'r') as file:
        return file.read().strip()

openai_api_key = get_openai_api_key()
os.environ["APIJESUE"] = openai_api_key

gpt4o_mini_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

planejador = Agent(
role="Content Planner",
goal="Plan engaging and factually accurate content on {topic}",
backstory="You’re working on planning a blog article " \
+ "about the topic: {topic}." \
+ "You collect information that helps the " \
+ "audience learn something " \
+ "and make informed decisions. " \
+ "Your work is the basis for " \
+ "the Content Writer to write an article on this topic.",
verbose=True,
allow_delegation=False,
max_rpm=2,
llm=gpt4o_mini_llm
)


escritor = Agent(
role="Content Writer",
goal="Write insightful and factually accurate " \
"opinion piece about the topic: {topic}",
backstory="You’re working on a writing " \
"a new opinion piece about the topic: {topic}. " \
"You base your writing on the work of " \
"the Content Planner, who provides an outline " \
"and relevant context about the topic. " \
"You follow the main objectives and " \
"direction of the outline, " \
"as provide by the Content Planner. " \
"You also provide objective and impartial insights " \
"and back them up with information " \
"provide by the Content Planner. " \
"You acknowledge in your opinion piece " \
"when your statements are opinions " \
"as opposed to objective statements.", \
verbose=True,
allow_delegation=False,
llm=gpt4o_mini_llm
)


editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory=(
        "You are an editor who receives a blog post from the Content Writer. "
        "Your goal is to review the blog post to ensure that it follows journalistic best practices, "
        "provides balanced viewpoints when providing opinions or assertions, "
        "and also avoids major controversial topics or opinions when possible."
    ),
    verbose=True,
    allow_delegation=False,
    llm=gpt4o_mini_llm
)


planejamento = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output=(
        "A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources."
    ),
    agent=planejador,
)


escrita = Task(
    description=(
        "1. Use the content plan to craft a compelling "
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand’s voice.\n"
    ),
    expected_output=(
        "A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs."
    ),
    agent=escritor,
)


edicao = Task(
    description=(
        "Proofread the given blog post for "
        "grammatical errors and "
        "alignment with the brand’s voice."
    ),
    expected_output=(
        "A well-written blog post in markdown format, in brazilian portuguese language,"
        "ready for publication, "
        "each section should have 2 or 3 paragraphs."
    ),
    agent=editor
)


crew = Crew(
    agents=[planejador, escritor, editor],
    tasks=[planejamento, escrita, edicao],
    verbose=2
)


result = crew.kickoff(inputs={"topic": "neural networks"})
