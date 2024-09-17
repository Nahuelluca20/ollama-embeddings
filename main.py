import ollama
import chromadb

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]


client = chromadb.Client()
collection = client.create_collection(name="docs")

for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

# an example prompt
prompt = "What animals are llamas related to?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)

# You can query the collection with a list of query texts, and Chroma will return the n most similar results.
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)

# from results dict get document and element with index [0][0]
data = results["documents"][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.1",
  prompt=f"Using this data: {data}. Responde to this prompt: {prompt}"
)

print(output['response'])