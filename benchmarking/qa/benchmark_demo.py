import openai

# replace the my_model_generate in eval.py
# set the base_url to the local vllm server, port 1234 as the default

def model_instruct_generate(messages, model_name, temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, max_tokens=1000):
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:1234/v1",
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def model_pretrain_generate(prompt, model_name, temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, max_tokens=1000):
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:1234/v1",
    )

    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )

    return response.choices[0].text

# Example 1: Instruct Generation
messages = [{"role": "user", "content": "What is your name?"}]
print(model_instruct_generate(messages, "med-llama3-8b"))

# Example 2: Pretrain Generation
prompt = "What is your name?"
print(model_pretrain_generate(prompt, "med-llama3-8b"))