#!/usr/bin/env python3
"""
Prometheus exporter for P2P cluster metrics.
Scrapes /health from P2P nodes and exposes Prometheus metrics.
"""

import json
import time
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Port mappings: port -> node name
TUNNEL_PORTS = {
    8770: 'lambda-h100',
    8771: 'lambda-gh200-a',
    8772: 'lambda-gh200-h',
    8773: 'lambda-gh200-a-2',
    8774: 'vast-4080s',
    8775: 'lambda-gh200-g',
    8776: 'lambda-gh200-i',
    8779: 'vast-3070',
    8780: 'vast-5090',
    8781: 'vast-4060ti',
    8782: 'vast-3060ti',
}

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/metrics':
            self.send_response(404)
            self.end_headers()
            return
            
        metrics = self.collect_metrics()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(metrics.encode('utf-8'))
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def collect_metrics(self):
        lines = []
        lines.append('# HELP ringrift_node_up Whether the P2P node is reachable')
        lines.append('# TYPE ringrift_node_up gauge')
        lines.append('# HELP ringrift_node_healthy Whether the P2P node reports healthy')
        lines.append('# TYPE ringrift_node_healthy gauge')
        lines.append('# HELP ringrift_disk_percent Disk usage percentage')
        lines.append('# TYPE ringrift_disk_percent gauge')
        lines.append('# HELP ringrift_memory_percent Memory usage percentage')
        lines.append('# TYPE ringrift_memory_percent gauge')
        lines.append('# HELP ringrift_cpu_percent CPU usage percentage')
        lines.append('# TYPE ringrift_cpu_percent gauge')
        lines.append('# HELP ringrift_selfplay_jobs Number of selfplay jobs running')
        lines.append('# TYPE ringrift_selfplay_jobs gauge')
        lines.append('# HELP ringrift_training_jobs Number of training jobs running')
        lines.append('# TYPE ringrift_training_jobs gauge')
        lines.append('# HELP ringrift_active_peers Number of active P2P peers')
        lines.append('# TYPE ringrift_active_peers gauge')
        
        for port, node in TUNNEL_PORTS.items():
            try:
                resp = requests.get(f'http://localhost:{port}/health', timeout=2)
                data = resp.json()
                up = 1
                healthy = 1 if data.get('healthy', False) else 0
                disk = data.get('disk_percent', 0)
                memory = data.get('memory_percent', 0)
                cpu = data.get('cpu_percent', 0)
                selfplay = data.get('selfplay_jobs', 0)
                training = data.get('training_jobs', 0)
                peers = data.get('active_peers', 0)
                role = data.get('role', 'unknown')
            except Exception:
                up = 0
                healthy = 0
                disk = memory = cpu = selfplay = training = peers = 0
                role = 'unknown'
            
            labels = f'node="{node}",port="{port}",role="{role}"'
            lines.append(f'ringrift_node_up{{{labels}}} {up}')
            lines.append(f'ringrift_node_healthy{{{labels}}} {healthy}')
            lines.append(f'ringrift_disk_percent{{{labels}}} {disk}')
            lines.append(f'ringrift_memory_percent{{{labels}}} {memory}')
            lines.append(f'ringrift_cpu_percent{{{labels}}} {cpu}')
            lines.append(f'ringrift_selfplay_jobs{{{labels}}} {selfplay}')
            lines.append(f'ringrift_training_jobs{{{labels}}} {training}')
            lines.append(f'ringrift_active_peers{{{labels}}} {peers}')
        
        return '\n'.join(lines) + '\n'

def main():
    port = 9094
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    print(f'Starting Prometheus P2P exporter on port {port}')
    server.serve_forever()

if __name__ == '__main__':
    main()
