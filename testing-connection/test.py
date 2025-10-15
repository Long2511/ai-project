from qdrant_client import QdrantClient, models

client = QdrantClient(url="https://8e62d30b-e86c-49d6-9f18-00cf52d45618.us-east4-0.gcp.cloud.qdrant.io:6333", 
                      api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SlUx9MpbVJJn8FoVOOrfVc6Nd39Pla1AqYwhgJ-bm0w")  

# create new collection "testing" 
# result a new collection on Qdrant server
client.create_collection(
    collection_name="testing",
    vectors_config={
        "dense": models.VectorParams(            # stage-1 pooled vector space
            size=768,
            distance=models.Distance.COSINE
        ),
        "colpali": models.VectorParams(     # token/patch multi-vector space
            size=128,
            distance=models.Distance.COSINE
        ),
    },
)

print(client.get_collection("testing"))
