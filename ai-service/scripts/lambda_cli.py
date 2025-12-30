#!/usr/bin/env python3
"""
Lambda Labs Cloud CLI - Provision and manage GPU instances.

Usage:
    python scripts/lambda_cli.py list          # List running instances
    python scripts/lambda_cli.py types         # Show available instance types
    python scripts/lambda_cli.py keys          # List SSH keys
    python scripts/lambda_cli.py launch <type> <region> <ssh_key> [name]
    python scripts/lambda_cli.py terminate <instance_id>

Environment:
    LAMBDA_API_KEY - Your Lambda Labs API key (or uses default from config)

Examples:
    python scripts/lambda_cli.py launch gpu_1x_gh200 us-east-3 ringrift-cluster ringrift-gh200-10
"""
import subprocess
import json
import sys
import os

# Default API key (can be overridden by env var)
DEFAULT_API_KEY = "secret_ringrift-lambda-cli_1997888713174c6b8ecdaf0dba4f3d9c.rHxT4chFYvWmjPn0PBEXOcPFkciuwQnu"
API_KEY = os.environ.get("LAMBDA_API_KEY", DEFAULT_API_KEY)
BASE_URL = "https://cloud.lambda.ai/api/v1"


def api_call(endpoint, method="GET", data=None):
    """Make an API call to Lambda Labs."""
    cmd = ["curl", "-s", "-u", f"{API_KEY}:", f"{BASE_URL}/{endpoint}"]
    if method == "POST":
        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json"])
        if data:
            cmd.extend(["-d", json.dumps(data)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout or result.stderr}


def list_instances():
    """List all running instances."""
    return api_call("instances").get("data", [])


def list_ssh_keys():
    """List available SSH keys."""
    return api_call("ssh-keys").get("data", [])


def get_instance_types():
    """Get available instance types with capacity."""
    return api_call("instance-types").get("data", {})


def launch_instance(instance_type, region, ssh_key_names, name=None, quantity=1):
    """Launch new instance(s)."""
    data = {
        "instance_type_name": instance_type,
        "region_name": region,
        "ssh_key_names": ssh_key_names,
        "quantity": quantity,
    }
    if name:
        data["name"] = name
    return api_call("instance-operations/launch", method="POST", data=data)


def terminate_instance(instance_ids):
    """Terminate instance(s)."""
    data = {"instance_ids": instance_ids if isinstance(instance_ids, list) else [instance_ids]}
    return api_call("instance-operations/terminate", method="POST", data=data)


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        instances = list_instances()
        print(f"Running instances ({len(instances)}):")
        for inst in instances:
            ip = inst.get('ip', 'pending...')
            print(f"  {inst['name']}: {ip} - {inst['instance_type']['name']} ({inst['status']})")
            print(f"    ID: {inst['id']}")
            print(f"    Region: {inst['region']['name']} ({inst['region']['description']})")

    elif cmd == "types":
        types = get_instance_types()
        print("Available instance types with capacity:")
        for name, info in sorted(types.items()):
            regions = info.get("regions_with_capacity_available", [])
            if regions:
                price = info["instance_type"]["price_cents_per_hour"] / 100
                specs = info["instance_type"]["specs"]
                print(f"  {name}: ${price:.2f}/hr")
                print(f"    GPUs: {specs['gpus']}, vCPUs: {specs['vcpus']}, RAM: {specs['memory_gib']}GB")
                print(f"    Regions: {[r['name'] for r in regions]}")

    elif cmd == "keys":
        keys = list_ssh_keys()
        print("SSH Keys:")
        for key in keys:
            print(f"  {key['name']} (id: {key['id']})")

    elif cmd == "launch":
        if len(sys.argv) < 5:
            print("Usage: lambda_cli.py launch <type> <region> <ssh_key> [name] [quantity]")
            print("\nExample: lambda_cli.py launch gpu_1x_gh200 us-east-3 ringrift-cluster my-instance")
            sys.exit(1)
        inst_type = sys.argv[2]
        region = sys.argv[3]
        ssh_key = sys.argv[4]
        name = sys.argv[5] if len(sys.argv) > 5 else None
        quantity = int(sys.argv[6]) if len(sys.argv) > 6 else 1
        
        print(f"Launching {quantity}x {inst_type} in {region}...")
        result = launch_instance(inst_type, region, [ssh_key], name, quantity)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        else:
            launched = result.get("data", {}).get("instance_ids", [])
            print(f"Successfully launched {len(launched)} instance(s):")
            for inst_id in launched:
                print(f"  - {inst_id}")
            print("\nNote: Instances take 1-2 minutes to become active.")
            print("Run 'python scripts/lambda_cli.py list' to check status.")

    elif cmd == "terminate":
        if len(sys.argv) < 3:
            print("Usage: lambda_cli.py terminate <instance_id> [instance_id2 ...]")
            sys.exit(1)
        instance_ids = sys.argv[2:]
        print(f"Terminating {len(instance_ids)} instance(s)...")
        result = terminate_instance(instance_ids)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        else:
            terminated = result.get("data", {}).get("terminated_instances", [])
            print(f"Terminated: {terminated}")

    elif cmd in ["-h", "--help", "help"]:
        print_usage()

    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
