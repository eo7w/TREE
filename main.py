import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests

app = FastAPI()

# تحديد المسار المطلق لمجلد templates لتجنب أخطاء السيرفر
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class AgentTaskRequest(BaseModel):
    agent_id: int
    prompt: str
    image_url: str = "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/run-agent")
async def run_agent_task(data: AgentTaskRequest):
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    api_key = os.getenv("NVIDIA_API_KEY", "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")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": data.prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": data.image_url
              }
            }
          ]
        }
      ],
      "model": "moonshotai/kimi-k3",
      "max_tokens": 1024,
      "seed": 0,
      "stream": False,
      "temperature": 1,
      "reasoning_effort": "max"
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        res_data = response.json()
        answer = res_data.get("choices", [{}])[0].get("message", {}).get("content", "تم تنفيذ المهمة بنجاح.")
        return {"status": "success", "agent_id": data.agent_id, "result": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
