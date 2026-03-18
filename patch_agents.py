import re
import os

agent_files = [
    "alibaba_qa_agent.py",
    "alibaba_orchestrator_agent.py",
    "alibaba_grounding_agent.py",
    "alibaba_spatial_agent.py",
    "alibaba_verifier_agent.py",
    "alibaba_logical_agent.py"
]

replacement = """
        import base64
        user_content = [{"type": "text", "text": prompt}]
        if blackboard.current_image_path and os.path.exists(blackboard.current_image_path):
            with open(blackboard.current_image_path, "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
"""

for file_name in agent_files:
    file_path = f"src/multi_agent/agents/{file_name}"
    with open(file_path, "r") as f:
        content = f.read()

    # 1. Update default model
    content = content.replace('model_name="qwen3-max"', 'model_name="qwen3-vl-plus"')

    # 2. Update messages array
    old_messages = '''        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]'''
    
    if old_messages in content:
        content = content.replace(old_messages, replacement.strip("\n"))
        print(f"Patched {file_name} successfully.")
    else:
        print(f"Failed to find match in {file_name}.")
    
    with open(file_path, "w") as f:
        f.write(content)
