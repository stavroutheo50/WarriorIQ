from __future__ import annotations

import os
import webbrowser
from threading import Timer

import uvicorn
from dotenv import load_dotenv


load_dotenv()


def server_config() -> tuple[str, int, bool]:
    """Return local defaults while honoring Render and OuiPanel ports."""
    render = os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"}
    hosted = render or "PORT" in os.environ or "SERVER_PORT" in os.environ
    port = int(os.getenv("SERVER_PORT") or os.getenv("PORT") or "8000")
    host = os.getenv("HOST", "0.0.0.0" if hosted else "127.0.0.1")
    open_browser = not hosted and host in {"127.0.0.1", "localhost"}
    return host, port, open_browser


def main():
    host, port, open_browser = server_config()
    local_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{local_host}:{port}"
    print("\nWarriorIQ")
    print("Local website:", url)
    print("Keep this terminal open while WarriorIQ is running.\n")
    if open_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.main:app", host=host, port=port, reload=False, proxy_headers=True)


if __name__ == "__main__":
    main()
