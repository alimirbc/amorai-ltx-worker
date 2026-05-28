import runpod
import json
import time
import os
import sys
import uuid
import subprocess
import base64
import urllib.request
import urllib.parse
import urllib.error

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = "8188"
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")

PROMPT_ENHANCER_MODEL = "/runpod-volume/models/prompt_enhancer/sulphur_prompt_enhancer_model-q8_0.gguf"
PROMPT_ENHANCER_MMPROJ = "/runpod-volume/models/prompt_enhancer/mmproj-BF16.gguf"

PROMPT_NODE_ID = "29"

comfyui_process = None
enhancer_llm = None


def start_comfyui():
    global comfyui_process
    print("[INFO] Starting ComfyUI...", flush=True)
    comfyui_process = subprocess.Popen(
        ["python3", "main.py", "--listen", COMFYUI_HOST, "--port", COMFYUI_PORT],
        cwd=COMFYUI_PATH,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def wait_for_comfyui(timeout=300):
    print("[INFO] Waiting for ComfyUI to be ready...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/system_stats", timeout=5
            ) as r:
                if r.status == 200:
                    print("[INFO] ComfyUI is ready.", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(2)
    print("[ERROR] ComfyUI did not start in time.", flush=True)
    return False


def load_enhancer():
    global enhancer_llm
    if not os.path.exists(PROMPT_ENHANCER_MODEL) or not os.path.exists(PROMPT_ENHANCER_MMPROJ):
        print("[prompt-enhancer] Model files not found — skipping.", flush=True)
        return
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import LlavaR11ChatHandler
        print("[prompt-enhancer] Loading model...", flush=True)
        chat_handler = LlavaR11ChatHandler(
            clip_model_path=PROMPT_ENHANCER_MMPROJ,
            verbose=False,
        )
        enhancer_llm = Llama(
            model_path=PROMPT_ENHANCER_MODEL,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("[prompt-enhancer] Ready.", flush=True)
    except Exception as e:
        print(f"[prompt-enhancer] Failed to load: {e}", flush=True)
        enhancer_llm = None


def enhance_prompt(prompt: str, image_b64: str | None = None) -> str:
    if enhancer_llm is None:
        return prompt
    try:
        if image_b64:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "You are a cinematic video prompt enhancer for the Sulphur-2 AI model. "
                        "Based on the image and the user's intent below, write a single detailed "
                        "video generation prompt describing the scene's visuals, lighting, motion, "
                        "and atmosphere. Output only the enhanced prompt, no extra commentary.\n\n"
                        f"User intent: {prompt}"
                    ),
                },
            ]
        else:
            content = (
                "You are a cinematic video prompt enhancer for the Sulphur-2 AI model. "
                "Rewrite the following as a detailed video generation prompt. "
                "Output only the enhanced prompt, no extra commentary.\n\n"
                f"User intent: {prompt}"
            )

        result = enhancer_llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=384,
            temperature=0.7,
        )
        enhanced = result["choices"][0]["message"]["content"].strip()
        print(f"[prompt-enhancer] {enhanced[:120]}...", flush=True)
        return enhanced
    except Exception as e:
        print(f"[prompt-enhancer] Enhancement failed, using original: {e}", flush=True)
        return prompt


def upload_image(image_b64: str, filename: str) -> str:
    image_bytes = base64.b64decode(image_b64)
    boundary = "----AmorAIBoundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())["name"]


def queue_workflow(workflow: dict) -> tuple[str, str]:
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            result = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[ERROR] ComfyUI /prompt rejected ({e.code}): {body}", flush=True)
        raise urllib.error.HTTPError(e.url, e.code, f"{e.reason} | body: {body}", e.headers, None)
    return result["prompt_id"], client_id


def poll_for_completion(prompt_id: str, timeout: int = 600) -> dict | None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ) as r:
                history = json.loads(r.read().decode())
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(3)
    return None


def collect_output_files(outputs: dict) -> list[str]:
    files = []
    for node_output in outputs.values():
        for key in ("videos", "images", "gifs"):
            for item in node_output.get(key, []):
                subfolder = item.get("subfolder", "")
                fname = item["filename"]
                path = os.path.join(COMFYUI_PATH, "output", subfolder, fname)
                if os.path.exists(path):
                    files.append(path)
    return files


def handler(job):
    job_input = job.get("input", {})
    workflow = job_input.get("workflow")
    if not workflow:
        return {"error": "No workflow provided in job input"}

    images = job_input.get("images", [])

    # --- Prompt enhancement ---
    # Extract the raw prompt from node 29, enhance it with the reference image
    raw_prompt = ""
    if PROMPT_NODE_ID in workflow:
        raw_prompt = workflow[PROMPT_NODE_ID].get("inputs", {}).get("value", "")

    if raw_prompt and enhancer_llm is not None:
        first_image_b64 = images[0]["image"] if images else None
        enhanced = enhance_prompt(raw_prompt, first_image_b64)
        workflow[PROMPT_NODE_ID]["inputs"]["value"] = enhanced

    # --- Upload reference images ---
    for img in images:
        name = img.get("name")
        data = img.get("image")
        if name and data:
            try:
                uploaded = upload_image(data, name)
                workflow_str = json.dumps(workflow).replace(
                    json.dumps(name)[1:-1], json.dumps(uploaded)[1:-1]
                )
                workflow = json.loads(workflow_str)
            except Exception as e:
                print(f"[WARN] Failed to upload image {name}: {e}", flush=True)

    print(f"[INFO] Queueing workflow...", flush=True)
    prompt_id, _ = queue_workflow(workflow)
    print(f"[INFO] Prompt ID: {prompt_id}", flush=True)

    history = poll_for_completion(prompt_id)
    if not history:
        return {"error": "Timeout waiting for workflow completion"}

    status = history.get("status", {})
    if status.get("status_str") == "error":
        messages = status.get("messages", [])
        return {"error": f"Workflow failed: {messages}"}

    output_files = collect_output_files(history.get("outputs", {}))
    if not output_files:
        return {"error": "Workflow completed but no output files found"}

    results = []
    for fpath in output_files:
        with open(fpath, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(fpath)[1].lower()
        file_type = "video" if ext in (".mp4", ".webm", ".avi", ".mov") else "image"
        results.append(
            {"filename": os.path.basename(fpath), "type": file_type, "data": data}
        )
        print(f"[INFO] Output: {os.path.basename(fpath)} ({file_type})", flush=True)

    return {"outputs": results}


if __name__ == "__main__":
    start_comfyui()
    if not wait_for_comfyui():
        sys.exit(1)

    load_enhancer()

    print("[INFO] Starting RunPod serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
