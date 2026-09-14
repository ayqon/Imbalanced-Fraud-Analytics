"""
Local HTTP Server runner for FraudGuard Web UI.
"""

import http.server
import socketserver
import os
import webbrowser
import sys

PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, format, *args):
        # Minimal logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def main():
    os.chdir(WEB_DIR)
    url = f"http://localhost:{PORT}"
    print(f"Starting FraudGuard Web Dashboard on {url}...")
    print("Press Ctrl+C to stop the server.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()


if __name__ == '__main__':
    main()
