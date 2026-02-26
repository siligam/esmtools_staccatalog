#!/usr/bin/env /albedo/soft/sw/spack-sw/python/3.11.7-gz3343q/bin/python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, HEAD')
        self.send_header('Access-Control-Allow-Headers', '*')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    # Serve from the catalog directory so /catalog.json is at the root
    catalog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalog')
    os.chdir(catalog_dir)
    test(CORSRequestHandler, HTTPServer, port=port)

