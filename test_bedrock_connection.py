from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", 
    region_name="us-east-1", 
    temperature=0.3
)

try:
    print("Connecting to Bedrock...")
    resp = llm.invoke("Say hello in one sentence.")
    print("\nResponse:")
    print(resp.content)
except Exception as e:
    print(f"\nError: {e}")

