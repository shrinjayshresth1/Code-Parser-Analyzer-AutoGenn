from autogen import AssistantAgent, UserProxyAgent

config = {
    "model": "gpt-3.5-turbo",
    "api_key": "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"
}

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=2,
    code_execution_config=False
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="""You are a code analysis expert. For any code:
    1. Generate AST in JSON
    2. Create Symbol Table
    3. Build Control Flow Graph
    4. Show Data Flow Graph
    5. Return ALL as one JSON""",
    llm_config={"config_list": [config]}
)

def analyze():
    print("Code Analyzer (Type 'quit' to exit)")
    while True:
        code = input("\nEnter code:\n")
        if code.lower() == 'quit':
            break
        
        user_proxy.initiate_chat(
            analyst,
            message=f"Analyze this code and return JSON:\n```{code}```"
        )

if __name__ == "__main__":
    analyze()