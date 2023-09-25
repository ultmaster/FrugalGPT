import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.core_skills import ConversationSummarySkill

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
# api_key, org_id = sk.openai_settings_from_dot_env()
# kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

# Alternative using Azure:
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service("dv", AzureChatCompletion(deployment, endpoint, api_key))

prompt = """Bot: How can I help you?
User: {{$input}}

---------------------------------------------

The intent of the user in 5 words or less: """

prompt_config = sk.PromptTemplateConfig(
    description="Gets the intent of the user.",
    type="completion",
    completion=sk.PromptTemplateConfig.CompletionConfig(0.0, 0.0, 0.0, 0.0, 500),
    input=sk.PromptTemplateConfig.InputConfig(
        parameters=[
            sk.PromptTemplateConfig.InputParameter(
                name="input", description="The user's request.", default_value=""
            )
        ]
    ),
)

prompt_template = sk.PromptTemplate(
    template=prompt,
    template_engine=kernel.prompt_template_engine,
    prompt_config=prompt_config,
)
function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)

get_intent = kernel.register_semantic_function(
    skill_name="OrchestratorPlugin",
    function_name="GetIntent",
    function_config=function_config,
)

async def main():
    result = await kernel.run_async(
        get_intent,
        input_str="I want to send an email to the marketing team celebrating their recent milestone.",
    )
    print(result)

asyncio.run(main())