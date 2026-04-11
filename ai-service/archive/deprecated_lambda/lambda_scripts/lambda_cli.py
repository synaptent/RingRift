#!/usr/bin/env python
"""Simple Lambda Cloud CLI for instance management."""
import argparse
import os
import sys

import requests
import yaml


def get_api_key():
    """Get API key from config file or environment."""
    # Try environment first
    if 'LAMBDA_API_KEY' in os.environ:
        return os.environ['LAMBDA_API_KEY']

    # Try config file
    config_path = os.path.expanduser('~/.lambda_cloud/config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            key = config.get('api_key')
            if key and key != 'YOUR_API_KEY_HERE':
                return key

    print("Error: No API key found. Set LAMBDA_API_KEY env var or edit ~/.lambda_cloud/config.yaml")
    sys.exit(1)

def api_request(endpoint, method='GET', data=None):
    """Make API request to Lambda Cloud."""
    api_key = get_api_key()
    url = f"https://cloud.lambdalabs.com/api/v1{endpoint}"
    headers = {'Authorization': f'Bearer {api_key}'}

    if method == 'GET':
        resp = requests.get(url, headers=headers)
    elif method == 'POST':
        resp = requests.post(url, headers=headers, json=data)
    elif method == 'DELETE':
        resp = requests.delete(url, headers=headers)

    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        return None

def list_instances():
    """List all running instances."""
    result = api_request('/instances')
    if result:
        instances = result.get('data', [])
        if not instances:
            print("No instances running")
            return

        print(f"{'ID':<40} {'Name':<20} {'Type':<15} {'Status':<10} {'IP'}")
        print('-' * 100)
        for i in instances:
            print(f"{i['id']:<40} {i.get('name', '-'):<20} {i['instance_type']['name']:<15} {i['status']:<10} {i.get('ip', '-')}")

def terminate_instance(instance_id):
    """Terminate an instance by ID."""
    result = api_request(f'/instances/{instance_id}', method='DELETE')
    if result:
        print(f"Terminated instance: {instance_id}")
    else:
        print(f"Failed to terminate instance: {instance_id}")

def main():
    parser = argparse.ArgumentParser(description='Lambda Cloud CLI')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('list', help='List instances')

    term = subparsers.add_parser('terminate', help='Terminate instance')
    term.add_argument('instance_id', help='Instance ID to terminate')

    args = parser.parse_args()

    if args.command == 'list':
        list_instances()
    elif args.command == 'terminate':
        terminate_instance(args.instance_id)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
