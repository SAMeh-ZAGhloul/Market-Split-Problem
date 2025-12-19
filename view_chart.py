#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import sys

# Change to the directory containing the HTML file
os.chdir('/Users/user/Downloads/Market Split Problem')

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    print("Open your browser and go to: http://localhost:8000/solve_time_chart.html")
    print("Press Ctrl+C to stop the server")
    
    try:
        webbrowser.open(f'http://localhost:{PORT}/solve_time_chart.html')
    except:
        pass
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
