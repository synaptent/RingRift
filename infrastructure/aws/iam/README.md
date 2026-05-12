# IAM policies — production EC2

## `ringrift-ses-sender-rotation-policy.json`

Lets the production EC2 instance role rotate the SES sender's own access keys
without needing root or a separate IAM admin user.

**Scope:** every action is pinned to the single resource
`arn:aws:iam::767371459652:user/ringrift-ses-sender`. The policy cannot be
abused to touch any other IAM principal.

**Actions allowed (all on `ringrift-ses-sender` only):**

| Action                     | Purpose                                      |
| -------------------------- | -------------------------------------------- |
| `iam:GetUser`              | sanity-check the user exists before mutating |
| `iam:ListAccessKeys`       | inventory current keys (max 2 per user)      |
| `iam:GetAccessKeyLastUsed` | confirm grace window after deactivation      |
| `iam:CreateAccessKey`      | mint replacement key                         |
| `iam:UpdateAccessKey`      | flip old key to `Inactive` after switchover  |
| `iam:DeleteAccessKey`      | retire old key after grace window            |

## Attaching to the EC2 instance role (recommended)

The production EC2 instance assumes role `RingRiftEC2SecretsRole`. Attaching
this policy there means rotation can run from the box itself with no shipped
credentials.

Run from any host that has IAM admin (e.g. AWS Console as root, CloudShell,
or a workstation profile with `iam:PutRolePolicy`):

```bash
aws iam put-role-policy \
  --role-name RingRiftEC2SecretsRole \
  --policy-name AllowSesSenderKeyRotation \
  --policy-document file://infrastructure/aws/iam/ringrift-ses-sender-rotation-policy.json
```

Verify:

```bash
aws iam get-role-policy \
  --role-name RingRiftEC2SecretsRole \
  --policy-name AllowSesSenderKeyRotation
```

## Rotation playbook (after attach)

SSH to EC2 (`ssh -i ~/.ssh/ringrift-staging-key.pem ubuntu@54.198.219.106`),
then:

```bash
# 1) Make sure we use the instance role, not the keys we're rotating
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1

# 2) Inventory
aws iam list-access-keys --user-name ringrift-ses-sender

# 3) Mint new key (capture into a tmp file with mode 600)
umask 077
NEW_JSON=$(aws iam create-access-key --user-name ringrift-ses-sender)
NEW_AKID=$(echo "$NEW_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["AccessKey"]["AccessKeyId"])')
NEW_SECRET=$(echo "$NEW_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["AccessKey"]["SecretAccessKey"])')

# 4) Patch ~/ringrift/.env (atomic, never echo secrets)
cd ~/ringrift
cp .env .env.bak.$(date +%s)   # remove this backup once rotation is verified
python3 - <<PYEOF
import os
new_id = os.environ["NEW_AKID"]
new_secret = os.environ["NEW_SECRET"]
with open(".env") as f: lines = f.readlines()
out = []
for line in lines:
    if line.startswith("AWS_ACCESS_KEY_ID="):
        out.append(f"AWS_ACCESS_KEY_ID={new_id}\n")
    elif line.startswith("AWS_SECRET_ACCESS_KEY="):
        out.append(f"AWS_SECRET_ACCESS_KEY={new_secret}\n")
    else:
        out.append(line)
with open(".env.tmp", "w") as f: f.writelines(out)
os.rename(".env.tmp", ".env")
os.chmod(".env", 0o600)
PYEOF
unset NEW_AKID NEW_SECRET NEW_JSON

# 5) Restart so ringrift-server picks up new keys
set -a; source .env; set +a
pm2 restart ringrift-server --update-env

# 6) Smoke test SES with the new keys
aws ses get-send-quota   # uses instance role; not the rotated keys

# Confirm the *application* sees the new key by hitting an endpoint that sends
# email, or by tailing logs:
pm2 logs ringrift-server --lines 50 --nostream

# 7) Capture the OLD AccessKeyId before deactivating
OLD_AKID=$(aws iam list-access-keys --user-name ringrift-ses-sender \
  --query 'AccessKeyMetadata[?Status==`Active` && AccessKeyId!=`'"$NEW_AKID"'`].AccessKeyId' \
  --output text)

# 8) Mark old key inactive (grace window — keeps it recoverable)
aws iam update-access-key \
  --user-name ringrift-ses-sender \
  --access-key-id "$OLD_AKID" \
  --status Inactive

# 9) After 10–60 minutes of green health, delete the old key
aws iam delete-access-key \
  --user-name ringrift-ses-sender \
  --access-key-id "$OLD_AKID"

# 10) Shred the env backup
shred -uz ~/ringrift/.env.bak.*
```

## Alternative: attach to a workstation user instead

If you'd rather rotate from a laptop than from EC2, attach the same policy
to that workstation's IAM user (e.g. `ringrift-cluster` on mac-studio):

```bash
aws iam put-user-policy \
  --user-name ringrift-cluster \
  --policy-name AllowSesSenderKeyRotation \
  --policy-document file://infrastructure/aws/iam/ringrift-ses-sender-rotation-policy.json
```

The trade-off: an admin-capable credential now lives on a laptop. The EC2
instance-role path keeps the privilege ephemeral and tied to that one host.

## Why this scope is safe

- No `iam:PassRole`, no policy mutation, no other-user access — the policy
  cannot be used to escalate privileges.
- AWS enforces a max of two access keys per user, so even an attacker with
  this exact policy could only swap the SES key, not exfiltrate IAM admin.
- Worst case if the EC2 itself is compromised: same blast radius as today
  (SES send at 1/sec, 200/day) plus the ability to rotate that single key.
  No new lateral movement is unlocked.
