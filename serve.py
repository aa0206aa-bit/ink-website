#!/usr/bin/env python3
# 開發 server：每個回應送 Cache-Control: no-cache，瀏覽器每次 revalidate（304 快），改檔即見新版
import http.server

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

if __name__ == "__main__":
    http.server.HTTPServer(("", 8123), NoCacheHandler).serve_forever()
