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

comfyui_process = None


def start_comfyui():
    global comfyui_process
    print("[INFO] Starting ComfyUI...", flush=True)
    comfyui_process = subprocess.Popen(
        ["python3", "main.py", "--listen", COMFYUI_HOST, "--port", COMFYUI_PORT],
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read().decode())
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

    print("[INFO] Starting RunPod serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
