# Release Instructions

This document describes how to create releases for RingRift.

## Pre-release Checklist

Before creating a release:

1. **Update version numbers**
   - `package.json` - main project version
   - `ai-service/pyproject.toml` - AI service version

2. **Update CHANGELOG.md**
   - Add new section for the release
   - List all changes since last release

3. **Run tests**

   ```bash
   npm test
   cd ai-service && pytest
   npm run test:orchestrator-parity
   ```

4. **Build and verify**
   ```bash
   npm run build
   ```

## Creating a GitHub Release

### Using GitHub CLI

```bash
# Tag the release
git tag -a v0.1.0-beta -m "First public beta release"
git push origin v0.1.0-beta

# Create the release
gh release create v0.1.0-beta \
  --title "v0.1.0-beta - First Public Beta" \
  --notes-file RELEASE_NOTES.md \
  --prerelease
```

### Using GitHub Web UI

1. Go to https://github.com/an0mium/RingRift/releases
2. Click "Draft a new release"
3. Create new tag: `v0.1.0-beta`
4. Set release title: "v0.1.0-beta - First Public Beta"
5. Check "Set as a pre-release"
6. Copy release notes from CHANGELOG.md
7. Click "Publish release"

## Release Notes Template

````markdown
## What's New

[Summary of major features]

## Installation

### Local Development

```bash
git clone https://github.com/an0mium/RingRift.git
cd ringrift
npm install
cp .env.example .env
docker-compose up -d
npm run dev
```
````

### Requirements

- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 14+ and Redis 6+ (or use Docker)

## Full Changelog

See [CHANGELOG.md](../../CHANGELOG.md) for complete list of changes.

## Known Issues

- No hosted demo (local setup required)
- Mobile UI not optimized
- AI response can be slow at highest levels

```

## Version Numbering

RingRift follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes to game rules or API
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes and improvements
- **PRE-RELEASE** (-alpha, -beta, -rc): Testing releases

## Release Branches

- `main` - Latest stable code
- `release/vX.Y.Z` - Release branches for maintenance
- `develop` - Integration branch (if using GitFlow)

## Post-release

After publishing:

1. Announce on relevant channels
2. Monitor for immediate issues
3. Update documentation if needed
4. Begin next development cycle
```
