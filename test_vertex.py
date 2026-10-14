from google import genai

# vertexai=True routes your requests to GCP and uses your ADC credentials + A$427 credit
client = genai.Client(
    vertexai=True,
    project="episodic-memory-benchmark",
    location="us-central1",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Confirm that Vertex AI is working with a brief greeting.",
)

print(response.text)
