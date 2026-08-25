from __future__ import annotations

import webbrowser
from threading import Timer

import uvicorn


def main():
    url = "http://127.0.0.1:8000"
    print("\nWarriorIQ")
    print("Local website:", url)
    print("Keep this terminal open while WarriorIQ is running.\n")
    Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
