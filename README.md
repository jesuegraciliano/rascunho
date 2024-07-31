# Rascunho

## Passo 1: informar a chave OpenAI

Para definir a chave, é preciso adicionar a chave OpenAI no arquivo `APIJESUE.txt`.

## Passo 2: instalar as dependências

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Passo 3: informar o assunto de interesse

No final do arquivo `blog.py`, mais especificamente na última linha, pode-se mudar o assunto de interesse. Exemplo:

```python
result = crew.kickoff(inputs={"topic": "neural networks"})
```

onde o valor de `topic` pode ser alterado.

## Passo 4: executar o programa

```sh
python blog.py
```
